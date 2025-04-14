##### RSRF environment #####
conda create -n iprf python=3.8 -y && conda activate iprf

# Pytorch installation is machine dependent, please install the correct version for your machine.
# https://pytorch.org/get-started/previous-versions/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# no module named 'svox2' or python setup.py on directory
pip install -e . --verbose

# no module named 'icecream'
conda install conda-forge::icecream -y

# no module named 'tqdm'
conda install conda-forge::tqdm -y

# no module named 'scipy'
conda install scipy -y

# no module named 'imageio'
conda install conda-forge::imageio -y

# no module named 'cv2'
conda install opencv -y

# no module named 'matplotlib'
conda install conda-forge::matplotlib -y

# no module named 'tensorboard'
conda install conda-forge::tensorboard -y

# no module named 'lpips'
conda install conda-forge::lpips -y

# for save images
conda install conda-forge::imageio-ffmpeg -y
