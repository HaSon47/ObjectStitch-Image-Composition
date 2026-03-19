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

def get_background(bg_path, loc_bbox):
    """
    input:
        bg_path: str - background image path
        loc_bbox: list [x1, y1, x2, y2] (xyxy)

    return:
        - background_img: PIL.Image (512, 512)
        - new_loc_bbox: list [x1, y1, x2, y2] in resized image
    """
    img = Image.open(bg_path).convert("RGB")

    return img, loc_bbox


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

def generate_image_batch(input):
    #bg_path = os.path.join(img_folder_path, 'ground_truth.jpg')
    bg_path = input["bg_path"]
    fg_path = bg_path
    fg_mask_path = input["exam_mask_path"]

    # get location bbox
    loc_bbox = input["loc_bbox"]
    # get examplar bbox
    exam_bbox = input["exam_bbox"]

    bg_img, bbox = get_background(bg_path, loc_bbox)
    bg_w, bg_h = bg_img.size

    bbox = expand_bbox_xyxy(bbox, bg_w, bg_h)
    exam_bbox = expand_bbox_xyxy(exam_bbox, bg_w, bg_h)
    bg_t       = sd_transform(bg_img)

    fg_img, fg_mask = get_foreground(fg_path, fg_mask_path, exam_bbox)
    black  = np.zeros_like(fg_mask)
    fg_mask= np.asarray(fg_mask)
    fg_img = np.asarray(fg_img)
    fg_img = np.where(fg_mask > 127, fg_img, black)
    fg_img = Image.fromarray(fg_img)

    fg_t       = clip_transform(fg_img)
    mask       = Image.fromarray(bbox2mask(bbox, bg_w, bg_h))
    mask_t     = mask_transform(mask)
    mask_t     = torch.where(mask_t > 0.5, 1, 0).float()
    inpaint_t  = bg_t * (1 - mask_t)
    bbox_t     = get_bbox_tensor(bbox, bg_w, bg_h)

    return {"bg_img":  inpaint_t.unsqueeze(0),
            "bg_mask": mask_t.unsqueeze(0),
            "fg_img":  fg_t.unsqueeze(0),
            "bbox":    bbox_t.unsqueeze(0),
            "origin_size": (bg_w, bg_h)
            }

def get_inputs(testdir, turn):
    inputs = []
    img_test_path = os.path.join(testdir, "Image")
    anno_class_test_path = os.path.join(testdir, "Anno")
    anno_bbox_test_path = os.path.join(testdir, "phase1_output_k3")
    exam_bbox_folder_path = os.path.join(testdir, "boxes_sam")
    mask_bbox_folder_path = os.path.join(testdir, "masks_sam")

    for file in tqdm(os.listdir(exam_bbox_folder_path)):
        # get turn_i and img_folder name
        anno_class_path = os.path.join(anno_class_test_path, file)
        with open(anno_class_path, 'r') as f:
            anno_class = json.load(f)
        origin_detailt = anno_class["origin"]
        turn_i = int(origin_detailt[-5])
        img_folder =  origin_detailt.split('/')[0]
        if turn_i != turn:
            continue

        # get exam bbox and exam mask
        exam_bbox_path = os.path.join(exam_bbox_folder_path, file)
        with open(exam_bbox_path, 'r') as f:
            anno_exam = json.load(f)
        exam_bboxes = anno_exam["boxes"]
        idx = random.randrange(len(exam_bboxes))
        exam_bbox = exam_bboxes[idx]
        exam_mask_path = os.path.join(mask_bbox_folder_path, file.split(".")[0], f"mask{idx}.png")

        # get img path
        img_path = os.path.join(img_test_path, file.replace(".json", '.png'))

        # get loc bbox
        loc_bbox_path = os.path.join(anno_bbox_test_path, file)
        with open(loc_bbox_path, 'r') as f:
            anno_loc = json.load(f)
        loc_bbox = anno_loc["pred_box"][0]

        input = {
            "bg_path": img_path,
            "loc_bbox": loc_bbox,
            "exam_bbox": exam_bbox,
            "exam_mask_path": exam_mask_path,
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
        "--ckpt_objstit",
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
        "--turn",
        type=int,
        default=1,
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
    
    inputs = get_inputs(opt.testdir, opt.turn)
    print(f'find {len(inputs)} pairs of test samples')
    log_txt = "/home/hachi/ObjectStitch-Image-Composition/Log/inference_location.txt"
    os.makedirs(os.path.dirname(log_txt), exist_ok=True)
    for input in tqdm(inputs):
        turn_i, img_folder = input["note"]
        try:
            batch = generate_image_batch(input)

            # nếu ảnh chỉ có 1 mask, thì next
            if not batch:
                continue

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
            comp_img = tensor2numpy(x_samples_ddim, image_size=(h,w))
            comp_img_2 = tensor2numpy(x_samples_ddim, image_size=img_size)
            # save composite results
            res_img_path = os.path.join(opt.outdir, f"Turn_{turn_i}", "img",  f"{img_folder}.png")
            os.makedirs(os.path.dirname(res_img_path), exist_ok=True)
            save_image_out(comp_img[-1], res_img_path)

            mask = batch["bg_mask"].squeeze(0)
            res_msk_path = os.path.join(opt.outdir, f"Turn_{turn_i}", "mask",  f"{img_folder}.png")
            os.makedirs(os.path.dirname(res_msk_path), exist_ok=True)
            save_mask_out(mask[-1], batch["origin_size"], res_msk_path)
            # for i in range(comp_img.shape[0]):
            #     if i > 0:
            #         res_path = os.path.join(opt.outdir, img_name.split('.')[0] + f'_sample{i}.png')
            #     else:
            #         res_path = os.path.join(opt.outdir, img_name.split('.')[0] + '.jpg')
            #     save_image(comp_img[i], res_path)
            #     print('save result to {}'.format(res_path))
            if not opt.skip_grid:
                grid_img  = generate_image_grid(batch, comp_img_2)
                grid_path = os.path.join(opt.outdir, f"Turn_{turn_i}", "grid",  f"{img_folder}.jpg")
                os.makedirs(os.path.dirname(grid_path), exist_ok=True)
                save_image_out(grid_img, grid_path)
        except Exception as e:
            with open(log_txt, 'a') as fout:
                fout.write(f"{input['bg_path'].split('/')[-1]}: {str(e)}\n")
            continue
