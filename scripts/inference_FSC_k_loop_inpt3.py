import argparse, os, sys, glob, json, random, math
from tqdm import tqdm
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
try:
    from lightning_fabric.utilities.seed import log
    log.propagate = False
except:
    pass
from torch import device
import torchvision

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(proj_dir)
sys.path.insert(0, proj_dir)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torchvision.transforms import Resize
from ldm.data.open_images import get_tensor, get_tensor_clip, get_bbox_tensor, bbox2mask, mask2bbox

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

def get_background(bg_path, all_bboxes, loc_bbox, exam_bbox):
    """
    input:
        bg_path: str - background image path
        all_bboxes: list<[x1, y1, x2, y2]> (xyxy)
        loc_bbox: [x1, y1, x2, y2] (xyxy)
        exam_bbox:[x1, y1, x2, y2] (xyxy)

    return:
        - background_img: PIL.Image (512, 512)
        - fit_bboxes_resize: list<[x1, y1, x2, y2]> in resized image
    """
    img = Image.open(bg_path).convert("RGB")
    W, H = img.size

    x1, y1, x2, y2 = loc_bbox
    assert x2 > x1 and y2 > y1, "Invalid bbox"

    # ---- Step 1: largest possible square ----
    square_size = min(W, H)

    # bbox center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # initial crop (centered at bbox)
    crop_x1 = int(cx - square_size / 2)
    crop_y1 = int(cy - square_size / 2)
    crop_x2 = crop_x1 + square_size
    crop_y2 = crop_y1 + square_size

    # ---- Step 2: clamp to image boundary ----
    if crop_x1 < 0:
        crop_x1 = 0
        crop_x2 = square_size
    if crop_y1 < 0:
        crop_y1 = 0
        crop_y2 = square_size
    if crop_x2 > W:
        crop_x2 = W
        crop_x1 = W - square_size
    if crop_y2 > H:
        crop_y2 = H
        crop_y1 = H - square_size

    # ---- Step 3: crop ----
    cropped_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    # ---- Step 4: bbox in cropped image ----
    new_x1 = x1 - crop_x1
    new_y1 = y1 - crop_y1
    new_x2 = x2 - crop_x1
    new_y2 = y2 - crop_y1

    # ---- Lấy các bbox còn lại vẫn nằm trong ảnh cropped image ----
    fit_bboxes = []
    for bbox in all_bboxes:
        if bbox != exam_bbox and bbox != loc_bbox:
        # if bbox != loc_bbox:
            b_x1, b_y1, b_x2, b_y2 = bbox
            new_b_x1 = b_x1 - crop_x1
            new_b_y1 = b_y1 - crop_y1
            new_b_x2 = b_x2 - crop_x1
            new_b_y2 = b_y2 - crop_y1

            if new_b_x1>=0 and new_b_y1>=0 and new_b_x2<=square_size and new_b_y2<=square_size:
                new_bbox = [new_b_x1, new_b_y1, new_b_x2, new_b_y2]
                fit_bboxes.append(new_bbox)

    # ---- Step 5: resize ----
    target_size = 512
    scale = target_size / square_size

    resized_img = cropped_img.resize(
        (target_size, target_size), Image.BILINEAR
    )

    new_loc_bbox = [
        int(new_x1 * scale),
        int(new_y1 * scale),
        int(new_x2 * scale),
        int(new_y2 * scale),
    ]

    fit_bboxes_resize = [new_loc_bbox]
    for bbox in fit_bboxes:
        b_x1, b_y1, b_x2, b_y2 = bbox
        new_fit_bbox = [
            int(b_x1 * scale),
            int(b_y1 * scale),
            int(b_x2 * scale),
            int(b_y2 * scale),
        ]
        fit_bboxes_resize.append(new_fit_bbox)

    return resized_img, fit_bboxes_resize

def get_k_bboxes(bboxes, k):
    """Random chọn k bbox trong bboxes (không lặp)."""
    k = min(k, len(bboxes))
    return random.sample(bboxes, k)

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


def generate_image_batch(img_folder_path, k):
    #bg_path = os.path.join(img_folder_path, 'ground_truth.jpg')
    bg_path = os.path.join(img_folder_path, 'inpainted_turn_3.png')
    fg_path = os.path.join(img_folder_path, 'ground_truth.jpg')
    fg_mask_path = os.path.join(img_folder_path, 'mask_4.png')
    anno_path = os.path.join(img_folder_path, 'fixed_annotation.json')
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    # chỉ lấy các trường hợp có đủ 4 mask obj
    if len(anno["inpainted_bboxes"])<4:
        return None
    
    # get location bbox
    loc_bbox = anno["inpainted_bboxes"][0]
    # get examplar bbox
    exam_bbox = anno["inpainted_bboxes"][3]
    all_bboxes = anno['inpainted_bboxes'][:3]

    bg_img, bboxes = get_background(bg_path, all_bboxes, loc_bbox, exam_bbox)
    bboxes = get_k_bboxes(bboxes, k) 
    bg_w, bg_h = bg_img.size
    bg_t       = sd_transform(bg_img)

    fg_img, fg_mask = get_foreground(fg_path, fg_mask_path, exam_bbox)
    black  = np.zeros_like(fg_mask)
    fg_mask= np.asarray(fg_mask)
    fg_img = np.asarray(fg_img)
    fg_img = np.where(fg_mask > 127, fg_img, black)
    fg_img = Image.fromarray(fg_img)

    fg_t       = clip_transform(fg_img)
    mask       = Image.fromarray(bbox2mask(bboxes[0], bg_w, bg_h))
    mask_t     = mask_transform(mask)
    mask_t     = torch.where(mask_t > 0.5, 1, 0).float()
    inpaint_t  = bg_t * (1 - mask_t)
    bbox_t     = get_bbox_tensor(bboxes[0], bg_w, bg_h)

    return {"bg_img":  inpaint_t.unsqueeze(0),
            "bg_mask": mask_t.unsqueeze(0),
            "fg_img":  fg_t.unsqueeze(0),
            "bbox":    bbox_t.unsqueeze(0),
            "bboxes": bboxes,
            "refference": bg_t.unsqueeze(0)
            }

def generate_image_batch_next_turn(batch, turn_i):
    # cần thay bg_img và bg_mask dựa trên bboxes[turn_i]
    bg_t = batch["bg_img"].squeeze(0)
    bbox = batch["bboxes"][turn_i]

    mask = Image.fromarray(bbox2mask(bbox, 512, 512))
    mask_t     = mask_transform(mask)
    mask_t     = torch.where(mask_t > 0.5, 1, 0).float()
    inpaint_t = bg_t * (1-mask_t)
    bbox_t     = get_bbox_tensor(bbox, 512, 512)

    batch["bg_img"] = inpaint_t.unsqueeze(0)
    batch["bg_mask"] = mask_t.unsqueeze(0)
    batch["bbox"] = bbox_t.unsqueeze(0)

    return batch

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

def save_image(img, img_path):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flag = cv2.imwrite(img_path, img)
    if not flag:
        print(img_path, img.shape)

def draw_bbox_on_background(image_nps, bboxes, color=(255,215,0), thickness=3):
    dst_list = []

    B = image_nps.shape[0]

    for i in range(B):
        img = image_nps[i].copy()

        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)

            img = cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)

        dst_list.append(img)

    dst_nps = np.stack(dst_list, axis=0)
    return dst_nps

def draw_bbox_on_background_turn(image_nps, bboxes, color=(255,215,0), thickness=3):
    dst_list = []

    B = image_nps.shape[0]

    for i in range(B):
        img = image_nps[i].copy()

        for box in bboxes[:i+1]:
            x1, y1, x2, y2 = map(int, box)

            img = cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)

        dst_list.append(img)

    dst_nps = np.stack(dst_list, axis=0)
    return dst_nps

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testdir",
        type=str,
        help="background image path",
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
        "--rootpath",
        type=str,
        help="FSC image path",
        default="./examples"
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
        "--seed",
        type=int,
        default=321,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=3,
        help="number of objects want to insert",
    )
    opt = parser.parse_args()
    return opt

def generate_image_grid(batch, comp_img):
    res_dict = {} 
    img_size = (512, 512)
    res_dict['bg']   = tensor2numpy(batch['bg_img'], image_size=img_size)
    res_dict['bbox'] = draw_bbox_on_background(res_dict['bg'], batch['bboxes'], color=(255,215,0), thickness=3)
    res_dict['fg']   = tensor2numpy(clip2sd(batch['fg_img']), image_size=img_size)
    res_dict['comp'] = comp_img
    x_border = (np.ones((img_size[0], 10, 3)) * 255).astype(np.uint8)
    grid_img  = []
    grid_row = [res_dict['bbox'][0], x_border, res_dict['fg'][0]]
    res_dict['comp'] = draw_bbox_on_background(res_dict['comp'], batch['bboxes'], color=(255,215,0), thickness=1)
    for i in range(res_dict['comp'].shape[0]):
        comp_img =  res_dict['comp'][i]
        grid_row += [x_border, comp_img]
    grid_img = np.concatenate(grid_row, axis=1)
    grid_img = Image.fromarray(grid_img)
    return grid_img

def generate_image_batch_k(batch, results):
    res_dict = {} 
    img_size = (512, 512)
    res_dict['bg'] = tensor2numpy(batch['refference'], image_size=img_size)
    res_dict['bbox'] = draw_bbox_on_background(res_dict['bg'], batch['bboxes'], color=(255,215,0), thickness=2)
    res_dict['fg']   = tensor2numpy(clip2sd(batch['fg_img']), image_size=img_size)
    res_dict['comp'] = np.array(results)
    x_border = (np.ones((img_size[0], 10, 3)) * 255).astype(np.uint8)
    grid_img  = []
    grid_row = [res_dict['bbox'][0], x_border, res_dict['fg'][0]]
    res_dict['comp'] = draw_bbox_on_background_turn(res_dict['comp'], batch['bboxes'], color=(255,215,0), thickness=2)
    for i in range(res_dict['comp'].shape[0]):
        comp_img =  res_dict['comp'][i]
        grid_row += [x_border, comp_img]
    grid_img = np.concatenate(grid_row, axis=1)
    grid_img = Image.fromarray(grid_img)
    return grid_img

def save_image_grid(list_img, boxes, path, color=(255, 215, 0), thickness=2, fill_alpha=0.3):
    """
    list_img: list[np.ndarray(H,W,3)]
    boxes: list[list[x1,y1,x2,y2]]
    path: save path

    fill_alpha: độ trong suốt của bbox fill ở ảnh đầu tiên
    """

    if len(list_img) == 0:
        raise ValueError("list_img is empty")

    drawn_imgs = []

    for img_idx, img in enumerate(list_img):

        img_draw = img.copy()

        # ---- Ảnh đầu tiên: fill rectangle ----
        if img_idx == 0:
            overlay = img_draw.copy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # fill

            # blend overlay với ảnh gốc
            img_draw = cv2.addWeighted(
                overlay, fill_alpha,
                img_draw, 1 - fill_alpha,
                0
            )

        # ---- Ảnh còn lại: outline ----
        else:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)

        drawn_imgs.append(img_draw)

    # ---- 1 hàng ngang ----
    N = len(drawn_imgs)
    h, w, c = drawn_imgs[0].shape

    grid_img = np.zeros((h, N * w, c), dtype=drawn_imgs[0].dtype)

    for idx, img in enumerate(drawn_imgs):
        grid_img[:, idx*w:(idx+1)*w] = img

    cv2.imwrite(path, grid_img)

if __name__ == "__main__":
    opt = argument_parse()
    weight_path = os.path.join(opt.ckpt_dir, "ObjectStitch", "ObjectStitch.pth")
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
    # num_samples  = opt.num_samples
    num_samples = 1
    guidance_scale = opt.scale
    if opt.fixed_code:
        seed_everything(opt.seed)
    start_code = torch.randn([num_samples]+shape, device=device)
    
    test_list = os.listdir(opt.testdir) #####
    num_case = len(test_list)
    print(f'find {len(test_list)} pairs of test samples')
    os.makedirs(opt.outdir, exist_ok=True)

    for idx, img_name in enumerate(test_list):

        img_folder_path = os.path.join(opt.rootpath, 'test', img_name.split('.')[0])
        if not os.path.exists(img_folder_path):
            img_folder_path = os.path.join(opt.rootpath, 'val', img_name.split('.')[0])
        
        batch = generate_image_batch(img_folder_path, opt.K)

        # nếu ảnh chỉ có 1 mask, thì next
        if not batch:
            continue
        
        results = []
        for turn_i in range(1, opt.K+1):
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
            comp_img = tensor2numpy(x_samples_ddim, image_size=img_size)
            results.append(comp_img[0])
            
            # chuẩn bị cho turn tiếp
            ## nếu turn_i > số bbox ảnh có => kết thúc
            if turn_i == len(batch['bboxes']):
                break
            batch["bg_img"] = x_samples_ddim
            batch = generate_image_batch_next_turn(batch, turn_i)

        # save composite results
        grid_img = generate_image_batch_k(batch, results)
        grid_path = os.path.join(opt.outdir, img_name.split('.')[0] + f'_grid.jpg')
        save_image(grid_img, grid_path)
        print(f'{idx}/{num_case} :save grid_result to {grid_path}')
        # if not opt.skip_grid:
        #     grid_img  = generate_image_grid(batch, comp_img)
        #     grid_path = os.path.join(opt.outdir, img_name.split('.')[0] + f'_grid.jpg')
        #     save_image(grid_img, grid_path)
        #     print('save grid_result to {}'.format(grid_path))

