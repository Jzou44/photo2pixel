import torch
import numpy as np
from PIL import Image


def convert_image_to_tensor(img):
    img = img.convert("RGB")
    img_np = np.array(img).astype(np.float32)
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]
    img_pt = torch.from_numpy(img_np)
    return img_pt


def convert_tensor_to_image(img_pt):
    img_pt = img_pt[0, ...].permute(1, 2, 0)
    result_rgb_np = img_pt.cpu().numpy().astype(np.uint8)
    return Image.fromarray(result_rgb_np)
