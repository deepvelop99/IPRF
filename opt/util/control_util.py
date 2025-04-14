import torch.nn.functional as F
import imageio
import numpy as np

# ckpt_weight : weighted sum is always 1
def weighted_sum(im1, im2, weight):
    im = weight * im1 + (1 - weight) * im2
    return im

def change_shape(tensor, target_shape=(756, 1008)):
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
    tensor = F.interpolate(tensor, size=target_shape, mode='bilinear', align_corners=False)
    tensor = tensor.squeeze(0).permute(1, 2, 0)  # (1, C, H, W) -> (H, W, C)
    return tensor

def save_image(tensor, filename):
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (1, C, H, W) -> (H, W, C)
    tensor = (tensor * 255).astype(np.uint8)
    imageio.imwrite(filename, tensor)