#!/bin/bash

SCENE=$1
STYLE=$2

data_type=llff
style_img=../data/styles/${STYLE}.jpg

ckpt_svox2=../opt/ckpt_svox2/${data_type}/${SCENE}
ckpt_rsrf_ctrl=../opt/ctrl/ckpts/${data_type}/${SCENE}_${STYLE}_ctrl

data_dir=../data/${data_type}/${SCENE}

if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    CUDA_VISIBLE_DEVICES=1 python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff.json
fi

if [[ ! -f "${ckpt_rsrf_ctrl}/only_albedo/ckpt.npz" ]]; then
    CUDA_VISIBLE_DEVICES='1, 0, 3, 2' python opt_style_ours_onlyalbedo.py -t ${ckpt_rsrf_ctrl}/only_albedo ${data_dir} \
                    -c configs/llff_fixgeom.json \
                    --init_ckpt ${ckpt_svox2}/ckpt.npz \
                    --style ${style_img} \
                    --mse_num_epoches 2 --nnfm_num_epoches 10 \
                    --content_weight 1e-3 \
                    --shading_weight 1e-1

    CUDA_VISIBLE_DEVICES='1, 0, 3, 2' python render_imgs.py ${ckpt_rsrf_ctrl}/only_albedo/ckpt.npz ${data_dir} \
                        --render_path --no_imsave
fi

if [[ ! -f "${ckpt_rsrf_ctrl}/only_shading/ckpt.npz" ]]; then
    CUDA_VISIBLE_DEVICES='1, 0, 3, 2' python opt_style_ours_onlyshading.py -t ${ckpt_rsrf_ctrl}/only_shading ${data_dir} \
                    -c configs/llff_fixgeom.json \
                    --init_ckpt ${ckpt_svox2}/ckpt.npz \
                    --style ${style_img} \
                    --mse_num_epoches 2 --nnfm_num_epoches 10 \
                    --content_weight 1e-3 \
                    --shading_weight 1e-1
    
    CUDA_VISIBLE_DEVICES='1, 0, 3, 2' python render_imgs.py ${ckpt_rsrf_ctrl}/only_shading/ckpt.npz ${data_dir} \
                        --render_path --no_imsave
fi