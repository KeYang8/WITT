# WITT: A Wireless Image Transmission Transformer For Semantic Communication

This repo holds the code for the paper:
WITT: A Wireless Image Transmission Transformer For Semantic Communications

# Proposed model

## Introduction
In this paper, we aim to redesign the vision Transformer (ViT) as a new backbone to realize semantic image transmission, termed wireless image transmission transformer (WITT). Previous works build upon convolutional neural networks (CNNs), which are inefficient in capturing global dependencies, resulting in degraded end-to-end transmission performance especially for high-resolution images. To tackle this, the proposed WITT employs Swin Transformers as a more capable backbone to extract long-range information. Different from ViTs in image classification tasks, WITT is highly optimized for image transmission while considering the effect of the wireless channel. Specifically, we propose a spatial modulation module to scale the latent representations according to channel state information, which enhances the ability of a single model to deal with various channel conditions. As a result, extensive experiments verify that our WITT attains better performance for different image resolutions, distortion metrics, and channel conditions. All pretrain models can be found in this link [https://pan.baidu.com/s/1X8gg7MKSZt-eDD4liALayw (passward:zujr)] and google drive link [https://drive.google.com/drive/folders/1YdnShbfIT03p_e30vjkV2wPKYOQPmUWp?usp=sharing].

![ ](overview.png)
Fig. 1. (a) The overall architecture of the proposed WITT scheme for wireless image transmission. (b) Two successive Swin Transformer Blocks. W-MSA and SW-MAS are multi-head self attention modules with regular and shifted windowing configurations, respectively.

## Experimental results
We show the examples of visual comparison under AWGN channel at SNR = 10dB. More experimental results show in [results](./results).

* we employ the BPG codec for compression combined with 5G LDPC codes for channel coding (marked as “BPG + LDPC”). Here, we considered 5G LDPC codes with a block length of 6144 bits for different coding rates and quadrature amplitude modulations (QAM). 
* the ideal capacity-achieving channel code is also considered during evaluation (marked as “BPG + Capacity”).

![ ](./results/visual_comparison_result.png)

# Installation
WITT supports python 3.8+ and PyTorch 1.9+

# Usage

## Train

* cbr = C/(2^(2i)*3*2), i denotes the downsample number. For CIFAR10, i=2; for HR_image, i=4.
* Pretrained model has no Channel ModNet module, is trained on a fixed channel state (SNR).

```
python train.py --training True --trainset {HR_image/CIFAR10} --testset {kodak/CLIC21/CIFAR10} -- distortion_metric {MSE/MS-SSIM} --pretrain {True/False} --channel_type {awgn/rayleigh} --C {8/16/24/32/48/64/96/128/192}
```
```
e.g.
python train.py --training True --trainset HR_image --testset kodak -- distortion_metric MSE --pretrain True --channel_type awgn --C 96
```
You can apply our method on your own images.

## Test
All pretrain models can be found in this link [https://pan.baidu.com/s/1X8gg7MKSZt-eDD4liALayw (passward:zujr)] and google drive link [https://drive.google.com/drive/folders/1YdnShbfIT03p_e30vjkV2wPKYOQPmUWp?usp=sharing].

```
python train.py --training False --trainset {HR_image/CIFAR10} --testset {kodak/CLIC21/CIFAR10} -- distortion_metric {MSE/MS-SSIM} --pretrain {True/False} --channel_type {awgn/rayleigh} --C {8/16/24/32/48/64/96/128/192}
```
```
e.g.
python train.py --training False --trainset HR_image --testset kodak -- distortion_metric MSE --pretrain False --channel_type awgn --C 96
```


