"""
Base on logs/turn_order.json to get order id, run 10 turns adding in annotated pairtally dataset
- crop into the inpainted region 
- use sam to product examplar mask
- use clip_t with threshold = 0.2405 to filter
"""
import argparse, os, sys, glob, json, random
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
from torch import device
from torchvision.utils import save_image
from transformers import CLIPProcessor, CLIPModel  # Thêm thư viện CLIP

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(proj_dir)
sys.path.insert(0, proj_dir)
# sys.path.append("/mnt/disk1/hachi/ImgEdit/code/ObjectStitch-Image-Composition")
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torchvision.transforms import Resize
from ldm.data.open_images import get_tensor, get_tensor_clip, get_bbox_tensor, bbox2mask, mask2bbox

# ==========================================
# KHỞI TẠO CLIP EVALUATOR MODEL
# ==========================================
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_id = "openai/clip-vit-base-patch32"
print(f"Loading Evaluator CLIP model '{clip_model_id}' on {clip_device}...")
clip_model_eval = CLIPModel.from_pretrained(clip_model_id).to(clip_device)
clip_processor_eval = CLIPProcessor.from_pretrained(clip_model_id)

def crop_object_by_mask(img: Image.Image, mask: Image.Image) -> Image.Image:
    """Cắt vùng vật thể dựa trên mask."""
    mask_np = np.array(mask.convert("L"))
    rows = np.any(mask_np > 128, axis=1)
    cols = np.any(mask_np > 128, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    cropped_img = img.crop((xmin, ymin, xmax, ymax))
    return cropped_img

def pad_to_square(img: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
    """Pad ảnh thành hình vuông."""
    width, height = img.size
    max_dim = max(width, height)
    square_img = Image.new("RGB", (max_dim, max_dim), fill_color)
    x_offset = (max_dim - width) // 2
    y_offset = (max_dim - height) // 2
    square_img.paste(img, (x_offset, y_offset))
    return square_img

def calculate_clip_score(image: Image.Image, text: str) -> float:
    """Tính điểm CLIP score."""
    inputs = clip_processor_eval(
        text=[text], 
        images=image, 
        return_tensors="pt", 
        padding=True
    ).to(clip_device)

    with torch.no_grad():
        outputs = clip_model_eval(**inputs)
        
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    
    cosine_sim = torch.matmul(image_embeds, text_embeds.t()).item()
    return cosine_sim

def filter_clip(full_img_np, mask_pil, class_name, threshold=0.2405):
    """
    Crop ảnh từ full_img_np bằng mask_pil, pad vuông và kiểm tra CLIP Score.
    """
    img_pil = Image.fromarray(full_img_np)
    
    # 1. Cắt vùng vật thể
    cropped_obj = crop_object_by_mask(img_pil, mask_pil)
    if not cropped_obj:
        return False, 0.0
        
    # 2. Pad vuông
    squared_obj = pad_to_square(cropped_obj, fill_color=(0, 0, 0))
    text_prompt = f"a photo of a {class_name}"
    
    # 3. Tính Score
    score = calculate_clip_score(squared_obj, text_prompt)
    
    if score >= threshold:
        return True, score
    return False, score

def clip2sd(x):
    # clip input tensor to  stable diffusion tensor
    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1,-1,1,1).to(x.device)
    STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1,-1,1,1).to(x.device)
    denorm = x * STD + MEAN
    sd_x = denorm * 2 - 1
    return sd_x

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def load_model_from_config(config, ckpt, verbose=False):
    print('load checkpoint {}'.format(ckpt))
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.eval()
    return model

clip_transform = get_tensor_clip(image_size=(224, 224))
sd_transform   = get_tensor(image_size=(512, 512))
mask_transform = get_tensor(normalize=False, image_size=(512, 512))

def get_background(bg_path, loc_bbox):
    """
    Crop vùng xung quanh loc_bbox với kích thước vuông max là 256.
    Trường hợp ảnh nhỏ hơn 256, lấy min của chiều rộng/chiều cao ảnh.
    """
    img = Image.open(bg_path).convert("RGB")
    img_w, img_h = img.size

    x1, y1, x2, y2 = loc_bbox

    # Kích thước hình vuông cần crop theo ý tưởng của bạn
    crop_size = min(256, min(img_w, img_h))

    # Lấy tâm của loc_bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Tính toán tọa độ crop ban đầu (đặt bbox vào giữa)
    crop_x1 = int(cx - crop_size / 2)
    crop_y1 = int(cy - crop_size / 2)
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size

    # Dịch chuyển crop box nếu bị tràn ra ngoài biên ảnh
    # Giữ nguyên kích thước crop_size
    if crop_x1 < 0:
        crop_x2 -= crop_x1  # Dịch sang phải
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1  # Dịch xuống dưới
        crop_y1 = 0
        
    if crop_x2 > img_w:
        crop_x1 -= (crop_x2 - img_w)  # Dịch sang trái
        crop_x2 = img_w
    if crop_y2 > img_h:
        crop_y1 -= (crop_y2 - img_h)  # Dịch lên trên
        crop_y2 = img_h

    # Ép kiểu và clamp cẩn thận lại lần cuối
    crop_x1 = max(0, int(crop_x1))
    crop_y1 = max(0, int(crop_y1))
    crop_x2 = min(img_w, int(crop_x2))
    crop_y2 = min(img_h, int(crop_y2))

    crop_box = [crop_x1, crop_y1, crop_x2, crop_y2]
    
    # Cắt ảnh background
    cropped_img = img.crop(crop_box)

    # Cập nhật tọa độ loc_bbox mới tương đối so với ảnh đã crop
    new_loc_bbox = [
        x1 - crop_x1,
        y1 - crop_y1,
        x2 - crop_x1,
        y2 - crop_y1
    ]

    return img, cropped_img, new_loc_bbox, crop_box


def get_foreground(fg_path, fg_mask_path, exam_bbox):
    """
    Crop fg_img và fg_mask_img theo exam_bbox, đảm bảo crop hình vuông.
    
    Args:
        fg_path (str): path tới ảnh foreground
        fg_mask_path (str): path tới ảnh mask tương ứng
        exam_bbox (tuple): (x_min, y_min, x_max, y_max)
    
    Returns:
        fg_crop (PIL.Image): foreground sau khi crop
        mask_crop (PIL.Image): mask sau khi crop
    """

    # Load images
    fg_img = Image.open(fg_path).convert("RGB")
    fg_mask = Image.open(fg_mask_path).convert("RGB")

    img_w, img_h = fg_img.size
    x_min, y_min, x_max, y_max = exam_bbox

    # Tính tâm bbox
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    # Kích thước hình vuông
    box_w = x_max - x_min
    box_h = y_max - y_min
    side = int(max(box_w, box_h))

    # Toạ độ crop vuông
    left   = int(cx - side / 2)
    top    = int(cy - side / 2)
    right  = left + side
    bottom = top + side

    # Clamp để không vượt biên ảnh
    left   = max(0, left)
    top    = max(0, top)
    right  = min(img_w, right)
    bottom = min(img_h, bottom)

    # Crop
    fg_crop = fg_img.crop((left, top, right, bottom))
    mask_crop = fg_mask.crop((left, top, right, bottom))

    return fg_crop, mask_crop

def expand_bbox_xyxy(bbox, img_width, img_height, scale=0.05):
    """
    Expand bbox (xyxy) by a ratio.

    Args:
        bbox: [x1, y1, x2, y2]
        img_width: image width
        img_height: image height
        scale: expansion ratio (0.1 = expand 10% each side)

    Returns:
        Expanded bbox in xyxy format
    """
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    dw = w * scale
    dh = h * scale

    new_x1 = max(0, x1 - dw)
    new_y1 = max(0, y1 - dh)
    new_x2 = min(img_width, x2 + dw)
    new_y2 = min(img_height, y2 + dh)

    return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

def generate_image_batch(input, exam_bbox, exam_mask_path):
    bg_path = input["bg_path"]
    fg_path = bg_path
    fg_mask_path = exam_mask_path

    loc_bbox = input["loc_bbox"]

    # Gọi hàm get_background mới
    original_img, bg_img, bbox, crop_box = get_background(bg_path, loc_bbox)
    bg_w, bg_h = bg_img.size
    orig_w, orig_h = original_img.size

    # Lưu mask ban đầu để tiện tính metric về sau
    origin_msk = Image.fromarray(bbox2mask([int(b) for b in loc_bbox], orig_w, orig_h))

    # Expand bbox
    bbox = expand_bbox_xyxy(bbox, bg_w, bg_h)
    exam_bbox = expand_bbox_xyxy(exam_bbox, orig_w, orig_h) # exam_bbox nằm trên ảnh gốc
    
    bg_t = sd_transform(bg_img)

    fg_img, fg_mask = get_foreground(fg_path, fg_mask_path, exam_bbox)
    black  = np.zeros_like(fg_mask)
    fg_mask= np.asarray(fg_mask)
    fg_img = np.asarray(fg_img)
    fg_img = np.where(fg_mask > 127, fg_img, black)
    fg_img = Image.fromarray(fg_img)
    fg_t = clip_transform(fg_img)
    
    mask = Image.fromarray(bbox2mask(bbox, bg_w, bg_h))
    mask_t = mask_transform(mask)
    mask_t = torch.where(mask_t > 0.5, 1, 0).float()
    
    inpaint_t = bg_t * (1 - mask_t)
    bbox_t = get_bbox_tensor(bbox, bg_w, bg_h)

    # Đưa original_img và crop_box vào kết quả trả về
    return {
        "original_img": original_img,
        "original_mask": origin_msk,
        "crop_box": crop_box,
        "bg_img":  inpaint_t.unsqueeze(0),
        "bg_mask": mask_t.unsqueeze(0),
        "fg_img":  fg_t.unsqueeze(0),
        "bbox":    bbox_t.unsqueeze(0),
        "origin_size": (bg_w, bg_h)
    }

def get_inputs(img_test_path, anno_test_path, msk_test_path,  order_log, turn_i):
    # các class sẽ loại
    weird_class = {
        'dental floss',
        'pen',
        'pen with cap',
        'pen without cap',
        'pencil',
        'plastic utensil',
        'straight plastic toothpick',
        'toothpick'
    }

    inputs = []

    with open(order_log, 'r') as f:
        img_turn = json.load(f)

    if turn_i == 0:
        for file in tqdm(os.listdir(img_test_path)):
            img_name = file.split('.')[0]
            img_path = os.path.join(img_test_path, file)
            
            anno_file = os.path.join(anno_test_path, f"{img_name}.json") 
            with open(anno_file, 'r') as f:
                anno = json.load(f)
            
            # get class
            class_name = anno["class_name"]
            if class_name in weird_class:
                continue
            
            for idex_order, order in enumerate(img_turn[img_name]):
                box_id = order[turn_i]

                loc_bbox = anno["loc_bbox"][box_id]
                exam_bbox_list = anno["exam_bbox"]
                exam_mask = os.path.join(msk_test_path, img_name)

                img_folder = f"{img_name}_{idex_order}"
                input = {
                    "bg_path": img_path,
                    "loc_bbox": loc_bbox,
                    "exam_bbox_list": exam_bbox_list,
                    "exam_mask": exam_mask,
                    "class_name": class_name,
                    "note": (turn_i, img_folder)
                }
                inputs.append(input)
    else:
        for file in tqdm(os.listdir(img_test_path)):
            img_name = file.split('.')[0]
            img_path = os.path.join(img_test_path, file)

            idex_order = int(img_name[-1])
            img_name = img_name[:-2]
            anno_file = os.path.join(anno_test_path, f"{img_name}.json") 
            with open(anno_file, 'r') as f:
                anno = json.load(f)
            
            # get class
            class_name = anno["class_name"]
            if class_name in weird_class:
                continue
            
            order = img_turn[img_name][idex_order]
            box_id = order[turn_i]

            loc_bbox = anno["loc_bbox"][box_id]
            exam_bbox_list = anno["exam_bbox"]
            exam_mask = os.path.join(msk_test_path, img_name)

            img_folder = f"{img_name}_{idex_order}"
            input = {
                "bg_path": img_path,
                "loc_bbox": loc_bbox,
                "exam_bbox_list": exam_bbox_list,
                "exam_mask": exam_mask,
                "class_name": class_name,
                "note": (turn_i, img_folder)
            }
            inputs.append(input)
    
    return inputs


def prepare_input(batch, model, shape, device, num_samples):
    if num_samples > 1:
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = torch.cat([batch[k]] * num_samples, dim=0)
    test_model_kwargs={}
    bg_img    = batch['bg_img'].to(device)
    bg_latent = model.encode_first_stage(bg_img)
    bg_latent = model.get_first_stage_encoding(bg_latent).detach()
    test_model_kwargs['bg_latent'] = bg_latent
    rs_mask = F.interpolate(batch['bg_mask'].to(device), shape[-2:])
    rs_mask = torch.where(rs_mask > 0.5, 1.0, 0.0)
    test_model_kwargs['bg_mask']  = rs_mask
    test_model_kwargs['bbox']  = batch['bbox'].to(device)
    fg_tensor = batch['fg_img'].to(device)
    
    c = model.get_learned_conditioning(fg_tensor)
    c = model.proj_out(c)
    uc = model.learnable_vector.repeat(c.shape[0], c.shape[1], 1) # 1,1,768
    return test_model_kwargs, c, uc

def tensor2numpy(image, normalized=False, image_size=(512, 512)):
    image = Resize(image_size, antialias=True)(image)
    if not normalized:
        image = (image + 1.0) / 2.0  # -1,1 -> 0,1; b,c,h,w
    image = torch.clamp(image, 0., 1.)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 2, 3, 1)
    image = image.numpy()
    image = (image * 255).astype(np.uint8)
    return image

def save_image_out(img,  img_path):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flag = cv2.imwrite(img_path, img)
    if not flag:
        print(img_path, img.shape)

def save_mask_out(mask, size, msk_path):
    """
    Args:
        mask: torch.Tensor (H,W) hoặc (1,H,W) hoặc (B,1,H,W)
        size: tuple (w, h)
        msk_path: str
    """

    # 1 detach + cpu
    mask = mask.detach().cpu()

    # 2 chuẩn hóa shape → [1,1,H,W]
    if mask.dim() == 2:          # (H,W)
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:        # (1,H,W)
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4:
        pass
    else:
        raise ValueError("Unsupported mask dimension")

    # 3 resize
    w, h = size
    mask = F.interpolate(
        mask.float(),
        size=(h, w),   # PyTorch dùng (H,W)
        mode="nearest" # segmentation mask → luôn dùng nearest
    )

    # 4 về lại shape [1,H,W]
    mask = mask.squeeze(0)

    # 5 đảm bảo giá trị 0–1
    if mask.max() > 1:
        mask = mask / 255.0

    # 6 save
    save_image(mask, msk_path)

def draw_bbox_on_background(image_nps, norm_bbox, color=(255,215,0), thickness=3):
    dst_list = []
    for i in range(image_nps.shape[0]):
        img = image_nps[i].copy()
        h,w,_ = img.shape
        x1 = int(norm_bbox[0,0] * w)
        y1 = int(norm_bbox[0,1] * h)
        x2 = int(norm_bbox[0,2] * w)
        y2 = int(norm_bbox[0,3] * h)
        dst = cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)
        dst_list.append(dst)
    dst_nps = np.stack(dst_list, axis=0)
    return dst_nps

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgdir",
        type=str,
        help="background image path",
        default="./examples"
    )
    parser.add_argument(
        "--annodir",
        type=str,
        default="./examples"
    )
    parser.add_argument(
        "--order_log",
        type=str,
        default="./examples"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="results"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="gpu id",
        default=0
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        type=bool,
        default=False,
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/v1.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="./checkpoints",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--ckpt_objstit",
        default="/mnt/disk1/hachi/ImgEdit/code/ObjectStitch-Image-Composition/checkpoints",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=321,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--num_turn",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--blend",
        action='store_true',
    )
    opt = parser.parse_args()
    return opt

def generate_image_grid(batch, comp_img):
    res_dict = {} 
    img_size = (512, 512)
    res_dict['bg']   = tensor2numpy(batch['bg_img'], image_size=img_size)
    res_dict['bbox'] = draw_bbox_on_background(res_dict['bg'], batch['bbox'], color=(255,215,0), thickness=3)
    res_dict['fg']   = tensor2numpy(clip2sd(batch['fg_img']), image_size=img_size)
    res_dict['comp'] = comp_img
    x_border = (np.ones((img_size[0], 10, 3)) * 255).astype(np.uint8)
    grid_img  = []
    grid_row = [res_dict['bbox'][0], x_border, res_dict['fg'][0]]
    res_dict['comp'] = draw_bbox_on_background(res_dict['comp'], batch['bbox'], color=(255,215,0), thickness=1)
    for i in range(comp_img.shape[0]):
        comp_img =  res_dict['comp'][i]
        grid_row += [x_border, comp_img]
    grid_img = np.concatenate(grid_row, axis=1)
    grid_img = Image.fromarray(grid_img)
    return grid_img


if __name__ == "__main__":
    opt = argument_parse()
    # weight_path = os.path.join(opt.ckpt_dir, "ObjectStitch.pth")
    weight_path = os.path.join(opt.ckpt_objstit)
    assert os.path.exists(weight_path), weight_path 
    config      = OmegaConf.load(opt.config)
    clip_path   = os.path.join(opt.ckpt_dir, 'openai-clip-vit-large-patch14')
    assert os.path.exists(clip_path), clip_path
    config.model.params.cond_stage_config.params.version = clip_path
    model       = load_model_from_config(config, weight_path)
    device      = torch.device(f'cuda:{opt.gpu}')
    model       = model.to(device)
    if opt.plms:
        print(f'Using PLMS samplers with {opt.sample_steps} sampling steps')
        sampler = PLMSSampler(model)
    else:
        print(f'Using DDIM samplers with {opt.sample_steps} sampling steps')
        sampler = DDIMSampler(model)

    img_size = (512,512)
    shape = [4, img_size[1]//8, img_size[0]//8]
    sample_steps = opt.sample_steps
    num_samples  = opt.num_samples
    guidance_scale = opt.scale
    if opt.fixed_code:
        seed_everything(opt.seed)
    start_code = torch.randn([num_samples]+shape, device=device)
    
    for turn_i in tqdm(range(opt.num_turn)):
        img_pass_score = {}
        log_pass_score_txt = os.path.join(opt.outdir, f"Turn_{turn_i}", "log_pass.txt")
        img_nopass_score = {}
        log_nopass_score_txt = os.path.join(opt.outdir, "nopass", f"Turn_{turn_i}", "log_nopass.txt")
        anno_test_path = os.path.join(opt.annodir, "Anno")
        msk_test_path = os.path.join(opt.annodir, "mask")
        if turn_i > 0:
            img_test_path = os.path.join(opt.outdir, f"Turn_{turn_i-1}", "img")
            inputs = get_inputs(img_test_path, anno_test_path, msk_test_path,  opt.order_log, turn_i)
        else:
            img_test_path = os.path.join(opt.imgdir, "Image")
            inputs = get_inputs(img_test_path, anno_test_path, msk_test_path,  opt.order_log, turn_i)
        print(f'find {len(inputs)} pairs of test samples for turn {turn_i}')
        for input in tqdm(inputs):
            turn_i, img_folder = input["note"]
            class_name = input["class_name"]
            for idx, exam_bbox in enumerate(input["exam_bbox_list"]):
                exam_mask_path = os.path.join(input["exam_mask"], f"mask{idx}.png")
                batch = generate_image_batch(input, exam_bbox, exam_mask_path)

                test_model_kwargs, c, uc = prepare_input(batch, model, shape, device, num_samples)
                samples_ddim, _ = sampler.sample(S=sample_steps,
                                                conditioning=c,
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False,
                                                eta=0.0,
                                                x_T=start_code,
                                                unconditional_guidance_scale=guidance_scale,
                                                unconditional_conditioning=uc,
                                                test_model_kwargs=test_model_kwargs)
                x_samples_ddim = model.decode_first_stage(samples_ddim[:,:4]).cpu().float()
                w, h = batch["origin_size"]
                # comp_img là kết quả inpaint cho vùng crop
                comp_img = tensor2numpy(x_samples_ddim, image_size=(h, w))
                comp_img_2 = tensor2numpy(x_samples_ddim, image_size=img_size)

                # Lấy thông tin ảnh gốc và tọa độ crop
                original_img_np = np.array(batch["original_img"])
                crop_x1, crop_y1, crop_x2, crop_y2 = batch["crop_box"]

                # ==========================================
                # ÁP DỤNG CLIP FILTER TRÊN ẢNH OUTPUT VUÔNG 512x512 (comp_img_2)
                # ==========================================
                original_mask = batch["original_mask"]
                mask_np = np.array(original_mask)
                
                # 1. Trích xuất mask ở đúng vùng crop_box
                crop_mask = mask_np[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Đảm bảo mask không rỗng
                if crop_mask.shape[0] > 0 and crop_mask.shape[1] > 0:
                    # 2. Resize mask này lên 512x512 để khớp với ảnh comp_img_2
                    crop_mask_512 = cv2.resize(crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                    mask_512_pil = Image.fromarray(crop_mask_512)
                    
                    # 3. Lấy ảnh vuông 512x512 (phần tử cuối cùng tương ứng với final_results[-1])
                    square_gen_img = comp_img_2[-1]
                    
                    # 4. Đánh giá CLIP
                    is_passed, score = filter_clip(square_gen_img, mask_512_pil, class_name, threshold=0.2405)
                    
                    # Nếu ảnh không pass threshold, skip luôn quá trình blend/paste tốn thời gian
                    if not is_passed:
                        if turn_i == 0:
                            img_nopass_score[img_folder] = score
                        else:
                            continue
                    else:
                        img_pass_score[img_folder] = score
                else:
                    continue # Nếu mask rỗng (lỗi), skip

                if opt.blend:
                    crop_h = crop_y2 - crop_y1
                    crop_w = crop_x2 - crop_x1
                    final_results = []
                    for i in range(comp_img.shape[0]):
                        res_full = original_img_np.copy()
                        
                        # 1. Trích xuất phần background gốc tại vùng crop
                        orig_crop = res_full[crop_y1:crop_y2, crop_x1:crop_x2]
                        gen_crop = comp_img[i]
                        
                        # Đảm bảo gen_crop có cùng size với orig_crop (tránh sai lệch nội suy)
                        if gen_crop.shape[:2] != (crop_h, crop_w):
                            gen_crop = cv2.resize(gen_crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
                            
                        # 2. Lấy mask inpaint từ batch (chính xác vùng model đã tác động)
                        # Batch shape thường là [B, 1, H, W]
                        mask_tensor = batch["bg_mask"][0].squeeze().cpu().numpy()
                        
                        # Resize mask về đúng kích thước crop
                        mask_np = cv2.resize(mask_tensor, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                        
                        # 3. Logic Blending với Gaussian Blur để feather biên
                        # Bạn có thể điều chỉnh kernel (21, 21) tùy theo độ lớn của crop_box
                        mask_blurred = cv2.GaussianBlur(mask_np * 255.0, (21, 21), 0) / 255.0
                        
                        # Mở rộng dimension để có thể nhân ma trận (broadcasting) với ảnh RGB
                        mask_np = mask_np[:, :, np.newaxis]
                        mask_blurred = mask_blurred[:, :, np.newaxis]
                        
                        # Kết hợp mask cứng và mask mờ
                        mask_final = 1.0 - (1.0 - mask_np) * (1.0 - mask_blurred)
                        
                        # Hòa trộn giữa ảnh gốc (vùng crop) và ảnh đã gen
                        blended_crop = orig_crop * (1.0 - mask_final) + gen_crop * mask_final
                        blended_crop = blended_crop.astype(np.uint8)
                        
                        # 4. Dán vùng đã blend trở lại ảnh gốc
                        res_full[crop_y1:crop_y2, crop_x1:crop_x2] = blended_crop
                        final_results.append(res_full)
                        
                    final_results = np.stack(final_results, axis=0)

                else:
                    # Dán ảnh inpaint trở lại ảnh gốc
                    final_results = []
                    for i in range(comp_img.shape[0]):
                        res_full = original_img_np.copy()
                        res_full[crop_y1:crop_y2, crop_x1:crop_x2] = comp_img[i]
                        final_results.append(res_full)
                        
                    final_results = np.stack(final_results, axis=0)

                if is_passed:
                    # Lưu kết quả ảnh cuối cùng đã được dán
                    res_img_path = os.path.join(opt.outdir, f"Turn_{turn_i}", "img",  f"{img_folder}.png")
                    os.makedirs(os.path.dirname(res_img_path), exist_ok=True)
                    save_image_out(final_results[-1], res_img_path)

                    # Lưu mask
                    res_msk_path = os.path.join(opt.outdir, f"Turn_{turn_i}", "mask",  f"{img_folder}.png")
                    os.makedirs(os.path.dirname(res_msk_path), exist_ok=True)
                    batch["original_mask"].save(res_msk_path)

                    if not opt.skip_grid:
                        # Grid lúc này sẽ show quá trình thực hiện trên vùng CROP (dễ debug hơn)
                        grid_img  = generate_image_grid(batch, comp_img_2)
                        grid_path = os.path.join(opt.outdir, f"Turn_{turn_i}", "grid",  f"{img_folder}.jpg")
                        os.makedirs(os.path.dirname(grid_path), exist_ok=True)
                        save_image_out(grid_img, grid_path)

                    # nếu đã có 1 mask thỏa mãn, tiếp tục với ảnh khác
                    break
                else:
                    # Chỉ lưu những cái ko pass với turn_0
                    # Lưu kết quả ảnh cuối cùng đã được dán
                    res_img_path = os.path.join(opt.outdir, "nopass", f"Turn_{turn_i}", "img",  f"{img_folder}.png")
                    os.makedirs(os.path.dirname(res_img_path), exist_ok=True)
                    save_image_out(final_results[-1], res_img_path)

                    # Lưu mask
                    res_msk_path = os.path.join(opt.outdir, "nopass", f"Turn_{turn_i}", "mask",  f"{img_folder}.png")
                    os.makedirs(os.path.dirname(res_msk_path), exist_ok=True)
                    batch["original_mask"].save(res_msk_path)

                    if not opt.skip_grid:
                        # Grid lúc này sẽ show quá trình thực hiện trên vùng CROP (dễ debug hơn)
                        grid_img  = generate_image_grid(batch, comp_img_2)
                        grid_path = os.path.join(opt.outdir,  "nopass", f"Turn_{turn_i}", "grid",  f"{img_folder}.jpg")
                        os.makedirs(os.path.dirname(grid_path), exist_ok=True)
                        save_image_out(grid_img, grid_path)

                    # nếu đã có 1 mask thỏa mãn, tiếp tục với ảnh khác
        
    
        # log pass img with score
        with open(log_pass_score_txt, 'w') as f:
            for img, score in img_pass_score.items():
                f.write(f"{img}\t{score}\n")
        
        # log pass img with score
        if turn_i == 0:
            with open(log_nopass_score_txt, 'w') as f:
                for img, score in img_nopass_score.items():
                    f.write(f"{img}\t{score}\n")
