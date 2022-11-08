# common library. There may be warnings for different versions.
# 常见库安装，版本无所谓，但是不同的版本运行可能会有警告（说不准哪个版本效果好，三张卡版本各不相同，精度各有利弊）
pip install numpy==1.16.0
pip install scikit-learn==0.21.3
pip install opencv-python
pip install tensorboard
pip install pycocotools
pip install pyyaml==3.12
pip install tensorboardX
pip install cython==0.27.3

# version of pytorch, and torchvision
# 1080ti, 2080ti 和 3090所用的各不相同，同样无法判别哪个版本更好。
# 命令不行使就单独下载，离线安装。 提供所用参考
1080ti: conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y && conda clean --all -y
2080ti: conda install pytorch torchvision cudatoolkit -c pytorch - y && conda clean --all -y
3090: conda install pytorch torchvision cudatoolkit -c pytorch - y && conda clean --all -y

# Please install the appropriate version of mmcv-full
# 我从来没有用下述命令安装成功过，只能离线安装，头疼。 同提供所用参考
pip --no-cache-dir install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html
1080ti: mmcv-full
2080ti: mmcv-full
3090: mmcv-full