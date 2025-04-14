DATA_TYPE=$1
SCENE=$2

ckpt_svox2=ckpt_svox2/${DATA_TYPE}/${SCENE}
data_dir=../data/${DATA_TYPE}/${SCENE}


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    CUDA_VISIBLE_DEVICES=1 python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff.json
fi

CUDA_VISIBLE_DEVICES=0 python render_imgs.py ${ckpt_svox2}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave