## Balancing Alignment and Uniformity (BAU) 

Official PyTorch implementation of *Generalizable Person Re-identification via Balancing Alignment and Uniformity* (NeurIPS 2024).
[[arXiv](https://arxiv.org/abs/2411.11471)] [[Project](https://sgvr.kaist.ac.kr/~yoonki/BAU/)]

**👋 Welcome to Simple and Easy-to-Use Codebase for Domain Generalizable Person Re-ID!**

## 🗓️ Updates
- [12/2024] Codes are released.
- [09/2024] BAU has been accepted to NeurIPS 2024.

## 📖 Overview
![overview](figs/overview.jpg)
> We propose a Balancing Alignment and Uniformity (BAU) framework, which effectively mitigates the polarized effect of data augmentation by maintaining a balance between alignment and uniformity. Specifically, BAU incorporates alignment and uniformity losses applied to both original and augmented images and integrates a weighting strategy to assess the reliability of augmented samples, further improving the alignment loss. Additionally, we introduce a domain-specific uniformity loss that promotes uniformity within each source domain, thereby enhancing the learning of domain-invariant features. Our BAU effectively exploits the advantages of data augmentation, which previous studies could not fully utilize, and achieves state-of-the-art performance without requiring complex training procedures.

## 🔨 Getting Started
### ● Installation
```shell
git clone https://github.com/yoonkicho/BAU.git
cd BAU
pip install -r requirements.txt
python setup.py develop
```
### ● Preparing Datasets
```shell
cd examples && mkdir data
```
Download re-ID datasets you need for your experiment, e.g.,  [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), [CUHK03](https://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html), and so on.
Make your data directory look like:
```
BAU/examples/data
├── Market-1501-v15.09.15
├── MSMT17
├── cuhk03-np
├── cuhk02
├── CUHK-SYSU
├── underground_grid
├── QMUL-iLIDS
├── prid_2011
└── VIPeR
```

## 📌 Training
We utilize two RTX-3090 GPUs for training. Please refer to `train.sh`.

**For M+MS+CS→C3 of Protocol-2:**
```
CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py \
-ds market1501 msmt17 cuhksysu -dt cuhk03  --iters 500 \
--logs-dir $PATH_FOR_LOGS\
```

**For M+MS+CS→C3 of Protocol-3:**
```
CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py \
-ds market1501dg msmt17dg cuhksysu -dt cuhk03  --iters 1000 \
--logs-dir $PATH_FOR_LOGS\
```

**For M+C2+C3+CS → GRID of Protocol-1:**
```
CUDA_VISIBLE_DEVICES=0,1 python examples/train_bau.py \
-ds market1501dg cuhk02dg cuhk03dg cuhksysu -dt grid --iters 500 \
--logs-dir $PATH_FOR_LOGS\
```

## 📌 Evaluation
We utilize a single RTX-3090 GPU for testing.

**For evaluation:**
```
CUDA_VISIBLE_DEVICES=0 python examples/test.py \
-d $TARGET_DATASET --resume $PATH_FOR_MODEL --logs-dir $PATH_FOR_LOGS\
```

## 🔗 Citation
If you find this code useful for your research, please consider citing our paper:

````BibTex
@inproceedings{
cho2024generalizable,
title={Generalizable Person Re-identification via Balancing Alignment and Uniformity},
author={Yoonki Cho and Jaeyoon Kim and Woo Jae Kim and Junsik Jung and Sung-eui Yoon},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=j25WK4GEGH}
}
````

## 🍀 Acknowledgement
Our code is based on [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) and [SpCL](https://github.com/yxgeee/SpCL), and the code for ViT backbones is from [TransReID](https://github.com/damo-cv/TransReID).
Thanks for their wonderful work!
