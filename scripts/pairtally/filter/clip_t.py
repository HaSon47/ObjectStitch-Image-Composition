import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# 1. Khởi tạo model và processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32" # Hoặc "openai/clip-vit-large-patch14" để kết quả tốt hơn

print(f"Loading CLIP model '{model_id}' on {device}...")
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

def crop_object_by_mask(img: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Hàm crop vùng vật thể dựa trên mask.
    Giả sử mask là ảnh grayscale, vùng trắng (hoặc > 0) là vật thể, vùng đen (0) là nền.
    """
    mask_np = np.array(mask.convert("L"))
    
    # Tìm các pixel thuộc về mask
    rows = np.any(mask_np > 128, axis=1)
    cols = np.any(mask_np > 128, axis=0)
    
    # Nếu mask rỗng (không có vật thể), trả về ảnh gốc hoặc một ảnh mặc định
    if not np.any(rows) or not np.any(cols):
        return None
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Cắt ảnh theo Bounding Box của mask
    cropped_img = img.crop((xmin, ymin, xmax, ymax))
    return cropped_img

def pad_to_square(img: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
    """
    Pad ảnh thành hình vuông bằng cách thêm viền (mặc định màu đen).
    Ảnh gốc sẽ được đặt ở chính giữa.
    """
    width, height = img.size
    max_dim = max(width, height)
    
    # Tạo một ảnh vuông mới với màu nền tùy chọn
    square_img = Image.new("RGB", (max_dim, max_dim), fill_color)
    
    # Tính toán vị trí x, y để dán ảnh gốc vào chính giữa
    x_offset = (max_dim - width) // 2
    y_offset = (max_dim - height) // 2
    
    square_img.paste(img, (x_offset, y_offset))
    return square_img

def calculate_clip_score(image: Image.Image, text: str) -> float:
    """
    Tính điểm CLIP score giữa ảnh và text.
    """
    inputs = processor(
        text=[text], 
        images=image, 
        return_tensors="pt", 
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        
    # Lấy image_embeds và text_embeds
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    
    # Chuẩn hóa (Normalize) các vector để tính Cosine Similarity
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    # Tính Cosine Similarity
    cosine_sim = torch.matmul(image_embeds, text_embeds.t()).item()
    return cosine_sim

def evaluate_clip_text_on_testset(test_data, root_path, folder_name, num_logs=5):
    total_score = 0.0
    valid_samples = 0
    scores_list = []
    
    img_folder_path = os.path.join(test_data, 'img')
    mask_folder_path = os.path.join(test_data, 'mask')
    
    # Tạo thư mục logs để lưu các mẫu ảnh test
    log_dir = os.path.join("logs", folder_name)
    os.makedirs(log_dir, exist_ok=True)
    saved_logs_count = 0

    for img_filename in tqdm(os.listdir(img_folder_path), desc="Evaluating CLIP Score"):
        img_path = os.path.join(img_folder_path, img_filename)
        mask_path = os.path.join(mask_folder_path, img_filename)

        # get class
        anno_path = os.path.join(root_path, f"{img_filename.split('.')[0][:-2]}.json")
        try:
            with open(anno_path, "r") as f:
                anno = json.load(f)
            class_name = anno["class_name"]
            
            # Đọc ảnh và mask
            img_obj = Image.open(img_path).convert("RGB")
            mask_obj = Image.open(mask_path).convert("L")

            # 1. Cắt vùng vật thể
            cropped_obj = crop_object_by_mask(img_obj, mask_obj)

            if not cropped_obj:
                continue
                
            # 2. Pad ảnh thành hình vuông
            squared_obj = pad_to_square(cropped_obj, fill_color=(0, 0, 0)) # Đổi fill_color=(255, 255, 255) nếu muốn nền trắng

            # Lựa chọn text prompt phù hợp
            text_prompt = f"a photo of a {class_name}"

            # 3. Tính CLIP Score với ảnh đã pad
            score = calculate_clip_score(squared_obj, text_prompt)
            
            # 4. Lưu một vài mẫu ra thư mục logs để kiểm tra
            if saved_logs_count < num_logs:
                save_path = os.path.join(log_dir, f"squared_{img_filename}")
                squared_obj.save(save_path)
                saved_logs_count += 1
            
            total_score += score
            valid_samples += 1
            scores_list.append((img_path, score))

        except Exception as e:
            print(f"Lỗi khi xử lý {img_path}: {e}")

    average_score = total_score / valid_samples if valid_samples > 0 else 0
    print(f"\n--- KẾT QUẢ ĐÁNH GIÁ ---")
    print(f"Tổng số mẫu hợp lệ: {valid_samples}")
    print(f"CLIP-text Score trung bình: {average_score:.4f}")
    
    return average_score, scores_list

# ==========================================
# CÁCH SỬ DỤNG
# ==========================================
if __name__ == "__main__":
    folder_list = [
        "Turn_0",
        # "Turn_2"
    ]
    l = []
    root_path = "/mnt/disk1/hachi/ImgEdit/data/PairTally/Anno"
    
    # 1. Khai báo tên file log định dạng txt
    log_file_path = "clip_scores_detail_log_squared_padding.txt"
    
    # 2. Mở file ở chế độ 'w' (write)
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        # Ghi Header cho file log (dùng \t để tạo khoảng cách Tab)
        log_file.write("Folder\tImage_Path\tCLIP_Score\n")
        log_file.write("-" * 100 + "\n") # Đường gạch ngang phân cách
        
        for folder in folder_list:
            data_path = f"/mnt/disk1/hachi/ImgEdit/Output/Pairtally/ObjStit_Finetune_crop256_maskSam_blend/{folder}"

            print(f"\nĐang đánh giá thư mục: {folder}")
            # Truyền folder vào để phân loại log ảnh vuông
            avg_score, detail_scores = evaluate_clip_text_on_testset(data_path, root_path, folder_name=folder, num_logs=10)
            l.append(avg_score)
            print(f"{folder}: {avg_score}")
            
            # 3. Ghi chi tiết từng ảnh vào file log
            for img_path, score in detail_scores:
                log_file.write(f"{folder}\t{os.path.basename(img_path)}\t{score:.4f}\n")
                
        print("------------------------------------")
        l_array = np.array(l)
        total_mean = l_array.mean()
        print(f"total: {total_mean}")
        
        # 4. Ghi phần tổng kết (Summary) ở cuối file
        log_file.write("\n" + "=" * 50 + "\n")
        log_file.write("SUMMARY\n")
        log_file.write("=" * 50 + "\n")
        log_file.write(f"Total Average Score:\t{total_mean:.4f}\n")
        
    print(f"\n✅ Đã lưu chi tiết điểm số vào file: {log_file_path}")
    print(f"✅ Các ảnh mẫu đã pad vuông được lưu tại thư mục: logs/")