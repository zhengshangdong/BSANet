# BSDN: Context-Aware Bidirectional Spatial Dropout Network for Weakly-Supervised Object Detection
This is a PyTorch implementation of our BSDN.

## Contents
### 1. Hardwares
NVIDIA GTX 1080Ti (11G of memory) is OK  
NVIDIA GTX 2080Ti (11G of memory) is OK  
NVIDIA GTX 3090Ti (24G of memory) is OK  
### 2. Installation
1. Clone the BSDN repository
`git`
See [install.sh](https://github.com/zhengshangdong/BSDN/blob/master/install.sh)
### 3. Training and testing


### 6.Known issues
1. Since the VOC2007 dataset is very small, the performance on VOC2007 is not stable. Please re-training our methods on VOC2012 or MS-COCO to verify the performance of BSDN. 
2. The best results of VOC datasets usually are not achieved in the last epoch. Please save the intermediate models. You can also set the random seed SEED. Note that setting SEED to a fixed value still cannot guarantee deterministic behavior.
