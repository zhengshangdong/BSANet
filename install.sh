# common library. There may be warnings for different versions.
pip install numpy==1.16.0
pip install scikit-learn==0.21.3
pip install opencv-python
pip install tensorboard
pip install pycocotools
pip install pyyaml==3.12
pip install tensorboardX
pip install cython==0.27.3

# version of pytorch, and torchvision
1080ti: conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y && conda clean --all -y
2080ti: conda install pytorch torchvision cudatoolkit -c pytorch - y && conda clean --all -y
3090: conda install pytorch torchvision cudatoolkit -c pytorch - y && conda clean --all -y

# Please install the appropriate version of mmcv-full
pip --no-cache-dir install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
