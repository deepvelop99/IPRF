# IPRF : Intrinsic-guided Photorealistic StyleTransfer for Radiance Fields

## Create Environment

### Unzip IPRF.zip
Unzip repository zip file
```
unzip IPRF.zip
```

### Install Dependencies

Install required dependencies and activate the environment :
* [Pytorch](https://pytorch.org/get-started/previous-versions/) installation is machine dependent, please install the correct version for your machine.
```
bash create_iprf_env.sh
pip install -e . --verbose
python setup.py
```

## Quick Start
### Input Data Preparation
IPRF supports datasets like ```NeRF-LLFF``` and Style data.
To quickly test the method, download a sample dataset:
* Place the downloaded data in ```IPRF/data/```.

### PIE-Net (Intrinsic Image Decomposition Extractor)

* PIE-Net for Intrinsic Image Decomposition:
* [PIE-Net](https://ivi.fnwi.uva.nl/cv/pienet/assets/PIE_NET_CVPR_2022_main_paper.pdf)
   * Read the [PIE-Net](https://ivi.fnwi.uva.nl/cv/pienet/assets/PIE_NET_CVPR_2022_main_paper.pdf)
   * Download the pre-trained model: [Pre-trained PIE-Net](https://uvaauas.figshare.com/articles/conference_contribution/real_world_model_t7/19940000)
   * Place the downloaded model in ```IPRF/controllable/iid_extractor/ckpt/<pre-trained model>```.

### Style Images
```
!pip intsall gdown
cd data
gdown 15V0PEXEyJK2YZi9RZLnpA1l2CZX6hGUv
unzip data.zip
cd ..
```

### 3D Interpolation
For interpolating between albedo and shading, use gradio :
```
cd IPRF/controllable/
python 3D.py
```

enjoy😊.
