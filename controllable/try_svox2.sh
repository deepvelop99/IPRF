#!/bin/bash 

DATA_TYPE=$1
SCENE=$2
ckpt_svox2=../opt/ckpt_svox2/${DATA_TYPE}/${SCENE}
data_dir=../data/${DATA_TYPE}/${SCENE}

if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    CUDA_VISIBLE_DEVICES=0 python ../opt/opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/${DATA_TYPE}.json
fi

if [[ ! -f "${ckpt_svox2}/test_renders_path/0040.png" ]]; then
    CUDA_VISIBLE_DEVICES=0 python ../opt/render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave
fi