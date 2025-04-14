SCENE=$1
STYLE=$2

data_type=llff
ckpt_svox2=ckpt_svox2/${data_type}/${SCENE}
ckpt_rsrf=ckpt_rsrf/${data_type}/${SCENE}_${STYLE}
data_dir=../data/${data_type}/${SCENE}
style_img=../data/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_svox2}/ckpt.npz" ]]; then
    CUDA_VISIBLE_DEVICES=2 python opt.py -t ${ckpt_svox2} ${data_dir} \
                    -c configs/llff.json
fi

CUDA_VISIBLE_DEVICES=1,0,3,2 python opt_style_ours.py -t ${ckpt_rsrf} ${data_dir} \
                -c configs/llff_fixgeom.json \
                --init_ckpt ${ckpt_svox2}/ckpt.npz \
                --style ${style_img} \
                --mse_num_epoches 2 --nnfm_num_epoches 10 \
                --content_weight 1e-3 \
                --shading_weight 1e-2
#                 --shading_weight 5e-2
# content weight : 0.001 \ shading weight : 0.05 : 5e-2, 0.01 : 1e-2 

CUDA_VISIBLE_DEVICES=2 python render_imgs.py ${ckpt_rsrf}/ckpt.npz ${data_dir} \
                    --render_path --no_imsave
