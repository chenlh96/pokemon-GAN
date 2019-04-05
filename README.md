# pokemon-GAN

A pratice project for me that aims to:

- Learn basic knowledge of deep learning
- Learn GAN
- Learn pytorch

## Project goal

The goal of the project itself is to generate new pokemon using GAN, to better understanding how the GAN can actually learn and generate, instead of learning to generate anime face, we would like to explore whether the GAN can generate the pokemon, the fictional creatures in pokemon series.

## Dataset

The dataset contains about 23000 pictures that are collected from the internet, since the kinds of the pokemon is only 800+, we then collect the pictures from the 3 different artists:

- Ken
- Dream work
- Anime official

Also, we do image augmentations for the images mentioned above to expand our dataset.

The image size we are going to test are: 64 * 64, 128 * 128 and 256 * 256

## architeture of gan

-DCGAN: mostly used in generating pokemon
-illustrationGAN: a famous GAN structure that used to generate anime face
-animeGAN: a GAN that used to generate high resolution anime gan, we found it in this [paper](https://arxiv.org/pdf/1708.05509.pdf), this GAN is based on DRFGAN, ResNet

## Implementation

pytorch, PIL, numpy