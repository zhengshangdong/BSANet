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
See [install.sh](https://github.com/zhengshangdong/BSDN/blob/master/install.sh)

3. Our method is built upon RINet ([RINet](https://github.com/XiaoxFeng/RINet)), you can also install RINet and replace the specifical files to run our BSANet.

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
Detection results will be dumped in the `Outputs/vgg16_voc2007/$model_path/test` folder. You can set `--vis` to `True` to visualize the detection results.  
Training your model:
```
CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py \
  --dataset voc2007 \
  --cfg configs/baselines/vgg16_voc2007.yaml \
  --bs 1
```
or
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train_net_step.py \
  --dataset voc2007 \
  --cfg configs/baselines/vgg16_voc2007.yaml \
  --bs 1
```
Note that the current implementation does not support multi-gpu testing and only support DataParallel (not DistributedDataParallel) based multi-gpu training. In our experiments, training a BSDN model needs ~12 hours, ~24 hours, and ~10 days for VOC 2007, 2012, and MS-COCO dataset on one GPU. When we use four 3090 GPUs, training a BSDN model on MS-COCO needs ~4.5 days. 

### 5.Known issues
1. Since the VOC2007 dataset is very small, the performance on VOC2007 is not stable. Please re-training our methods on VOC2012 or MS-COCO to verify the performance of BSDN. 
2. The best results of VOC datasets usually are not achieved in the last epoch. Please save the intermediate models. You can also set the random seed SEED. Note that setting SEED to a fixed value still cannot guarantee deterministic behavior.
