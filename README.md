# WITT: A Wireless Image Transmission Transformer For Semantic Communication

This repo holds the code for the paper:
WITT: A Wireless Image Transmission Transformer For Semantic Communications

# Proposed model

## Introduction
In this paper, we aim to redesign the vision Transformer (ViT) as a new backbone to realize semantic image transmission, termed wireless image transmission transformer (WITT). Previous works build upon convolutional neural networks (CNNs), which are inefficient in capturing global dependencies, resulting in degraded end-to-end transmission performance especially for high-resolution images. To tackle this, the proposed WITT employs Swin Transformers as a more capable backbone to extract long-range information. Different from ViTs in image classification tasks, WITT is highly optimized for image transmission while considering the effect of the wireless channel. Specifically, we propose a spatial modulation module to scale the latent representations according to channel state information, which enhances the ability of a single model to deal with various channel conditions. As a result, extensive experiments verify that our WITT attains better performance for different image resolutions, distortion metrics, and channel conditions. All pretrain models can be found in this link [] and google drive link [].

![ ](overview.png)
Fig. 1. (a) The overall architecture of the proposed WITT scheme for wireless image transmission. (b) Two successive Swin Transformer Blocks. W-MSA and SW-MAS are multi-head self attention modules with regular and shifted windowing configurations, respectively.

## Experimental results


# Installation
WITT supports python 3.8+ and PyTorch 1.9+

# Usage


