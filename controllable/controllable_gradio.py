import torch
import torchvision
import torch.nn.functional as F
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap
from util import config_util

import imageio
import cv2
from tqdm import tqdm
import pdb
from icecream import ic
import copy
import time

from util.control_util import weighted_sum

import cv2
import numpy as np
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.cuda.empty_cache()

device = 'cuda'
parser = argparse.ArgumentParser()

# Add two checkpoint
parser.add_argument('ckpt_albedo', type=str, help='Path to the albedo checkpoint')
parser.add_argument('ckpt_shading', type=str, help='Path to the shading checkpoint')

# Add results dir
parser.add_argument('results_dir', type=str, help='Path to the 3D style interpolation results')

# Add ckpt weight for ckpt balancing
parser.add_argument('--albedo_weight', type=float, default=5e-1, help='weight ckpt')
parser.add_argument('--shading_weight', type=float, default=5e-1, help='weight ckpt')

config_util.define_common_args(parser)

parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--train', action='store_true', default=False, help='render train set')
parser.add_argument('--render_path',
                    action='store_true',
                    default=False,
                    help="Render path instead of test images (no metrics will be given)")
parser.add_argument('--timing',
                    action='store_true',
                    default=False,
                    help="Run only for timing (do not save images or use LPIPS/SSIM; "
                    "still computes PSNR to make sure images are being generated)")
parser.add_argument('--no_vid',
                    action='store_true',
                    default=False,
                    help="Disable video generation")
parser.add_argument('--no_imsave',
                    action='store_true',
                    default=False,
                    help="Disable image saving (can still save video; MUCH faster)")
parser.add_argument('--fps',
                    type=int,
                    default=30,
                    help="FPS of video")

# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")

# Random debugging features
parser.add_argument('--ray_len',
                    action='store_true',
                    default=False,
                    help="Render the ray lengths")

args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)

albedo_weight = args.albedo_weight
shading_weight = args.shading_weight

# render_directory path
render_dir = args.results_dir
want_metrics = True
if args.render_path:
    assert not args.train
    render_dir += '3d_interpolate'
    want_metrics = False
print(args.data_dir)
dset = datasets[args.dataset_type](args.data_dir, split="test_train" if args.train else "test",
                                    **config_util.build_data_options(args))

# ckpt grid load
albedo_styled_grid = svox2.SparseGrid.load(args.ckpt_albedo, device=device)
shading_styled_grid = svox2.SparseGrid.load(args.ckpt_shading, device=device)

# 3d interpolation # 값 치환으로 충분할 듯
albedo_styled_grid.sh_data.data = albedo_weight * albedo_styled_grid.sh_data.data + shading_weight * shading_styled_grid.sh_data.data

config_util.setup_render_opts(albedo_styled_grid.opt, args)
config_util.setup_render_opts(shading_styled_grid.opt, args)

os.makedirs(render_dir, exist_ok=True)

start = time.time()
with torch.no_grad():
    n_images = dset.render_c2w.size(0) if args.render_path else dset.n_images
    img_eval_interval = max(n_images // args.n_eval, 1)
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    n_images_gen = 0
    c2ws = dset.render_c2w.to(device=device) if args.render_path else dset.c2w.to(device=device)

    frames = []
    
    shading_sub = []
    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        dset_h, dset_w = dset.get_image_size(img_id)
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', img_id),
                           dset.intrins.get('fy', img_id),
                           dset.intrins.get('cx', img_id) + (w - dset_w) * 0.5,
                           dset.intrins.get('cy', img_id) + (h - dset_h) * 0.5,
                           w, h,
                           ndc_coeffs=dset.ndc_coeffs)

        # Image Rendering on grid1 (ckpt1)
        im1 = albedo_styled_grid.volume_render_image(cam, use_kernel=True, return_raylen=args.ray_len)
        im1.clamp_(0.0, 1.0)

        im2 = shading_styled_grid.volume_render_image(cam, use_kernel=True, return_raylen=args.ray_len)
        im2.clamp_(0.0, 1.0)

        im = im1 # 3d interpolation

        # for test
        if not args.render_path:
            im_gt = dset.gt[img_id].to(device=device)
            mse = (im - im_gt) ** 2
            mse_num : float = mse.mean().item()
            psnr = -10.0 * math.log10(mse_num)
            avg_psnr += psnr
            if not args.timing:
                ssim = compute_ssim(im_gt, im).item()
                avg_ssim += ssim
                if not args.no_lpips:
                    lpips_i = lpips_vgg(im_gt.permute([2, 0, 1]).contiguous(),
                            im.permute([2, 0, 1]).contiguous(), normalize=True).item()
                    avg_lpips += lpips_i
                    print(img_id, 'PSNR', psnr, 'SSIM', ssim, 'LPIPS', lpips_i)
                else:
                    print(img_id, 'PSNR', psnr, 'SSIM', ssim)
        img_path = path.join(render_dir, f'{img_id:04d}.png')
        im = im.cpu().numpy()
        if not args.render_path:
            im_gt = dset.gt[img_id].numpy()
            im = np.concatenate([im_gt, im], axis=1)
        if not args.timing:
            im = (im * 255).astype(np.uint8)
            if not args.no_imsave:
                imageio.imwrite(img_path, im)
            if not args.no_vid:
                frames.append(im)
        im = None
        n_images_gen += 1
    end = time.time()
    print(end - start)

    if want_metrics:
        print('AVERAGES')

        avg_psnr /= n_images_gen
        with open(path.join(render_dir, 'psnr.txt'), 'w') as f:
            f.write(str(avg_psnr))
        print('PSNR:', avg_psnr)
        if not args.timing:
            avg_ssim /= n_images_gen
            print('SSIM:', avg_ssim)
            with open(path.join(render_dir, 'ssim.txt'), 'w') as f:
                f.write(str(avg_ssim))
            if not args.no_lpips:
                avg_lpips /= n_images_gen
                print('LPIPS:', avg_lpips)
                with open(path.join(render_dir, 'lpips.txt'), 'w') as f:
                    f.write(str(avg_lpips))
    if not args.no_vid and len(frames):
        vid_path = render_dir + '.mp4'
        imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)  # pip install imageio-ffmpeg
        for i, frame in enumerate(frames):
            vid_path = render_dir + '/{0:04d}'.format(i) + '.png' 
            imageio.imwrite(vid_path, frame)  # pip install imageio-ffmpeg