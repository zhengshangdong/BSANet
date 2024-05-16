# BSANet: Data-Driven Bidirectional Spatial-Adaptive Network for Weakly Supervised Object Detection in Remote Sensing Images
This is a PyTorch implementation of our BSANet.

## Contents
### 1. Hardwares
NVIDIA GTX 1080Ti, 2080Ti, 3090 are OK  

### 2. Installation
1. Clone the BSANet repository  
```
git clone https://github.com/zhengshangdong/BSANet.git
```
2. Install libraries  
See `requirements.yaml`

cd BSANet/lib

bash make.sh



3. Our method is built upon [RINet](https://github.com/XiaoxFeng/RINet), you can also install RINet and replace the specifical files to run our BSANet.

### 3. Datasets  
Download NWPU and DIOR datasets and use the following basic structure to organize these data
```
$NWPUV2/                           
$VOC2007/Annotations
$VOC2007/JPEGImages
$VOC2007/ImageSets
```
```
$DIOR/                           
$VOC2007/Annotations
$VOC2007/ImageSets
$VOC2007/JPEGImages
```
All the datasets can be downloaded from [NWPU](https://drive.google.com/file/d/15xd4TASVAC2irRf02GA4LqYFbH7QITR-/view?usp=sharing) and [DIOR](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC). 

Download selective search proposals from [NWPU](https://drive.google.com/file/d/1VnmUDPomgTgmHvH3CemFOIWTLuVR5f-t/view?usp=sharing) and [DIOR](https://drive.google.com/file/d/1wbivkAxqBQB4vAX0APmVzIOhuawHpsPV/view?usp=sharing), and put it in the data/selective_search_data/

Download pretrained ImageNet weights from [here](https://drive.google.com/drive/folders/0B1_fAEgxdnvJSmF3YUlZcHFqWTQ), and put it in the data/imagenet_weights/

### 4. Training and testing
Evaluating the released model:
```
# training
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc vgg16

# testing for mAP
./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16

# testing for CorLoc
./experiments/scripts/test_faster_rcnn_corloc.sh 0 pascal_voc vgg16
```
Please change `ITERS` in the aforementioned `.sh` files to train and test the correct models.

### 5.Downloading models and modifying the corresponding
Dowload NWPU [T=7](www.baidu.com) and put it in the output/vgg16/voc_2007_trainval/default/. Please modify `T=7` in `FFE.py`, `self._classes` for NWPU in `pascal_voc.py`

Dowload DIOR [T=1](www.baidu.com) and put it in the output/vgg16/voc_2007_trainval/default/. Please modify `T=1` in `FFE.py`, `self._classes` for DIOR in `pascal_voc.py`


### 6.Acknowledgement
We borrowed code from [RINet](https://github.com/XiaoxFeng/RINet). Thanks so much for this excellent work.

