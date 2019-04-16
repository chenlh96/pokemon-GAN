# pokemon-GAN

A pratice project for me that aims to:

- Learn basic knowledge of deep learning
- Learn GAN
- Learn pytorch

## Project goal

The goal of the project itself is to generate new pokemon using GAN, to better understanding how the GAN can actually learn and generate, instead of learning to generate anime face, we would like to explore whether the GAN can generate the pokemon, the fictional creatures in pokemon series.

## Dataset

We use the following 3 datasets to test the performance of different GANs

### pokemon dataset

This is the main dataset we try to train the GAN since we hope that new pokemon can be generate through the model. The dataset contains about 23000 pictures that are collected from the internet, since the kinds of the pokemon is only 800+, we then collect the pictures from the 3 different artists:

- Ken
- Dream work
- Anime official

Also, we do image augmentations for the images mentioned above to expand our dataset.

The image size we are going to test are: 64 * 64, 128 * 128 and 256 * 256

### Fashion MNIST

This dataset can be obtain by pytorch

### anime face dataset

The dowmload site is [here](https://github.com/jayleicn/animeGAN). Thanks @jayleicn to clean the raw data and put it on the cloud.

## architeture of gan

- DCGAN: mostly used in generating pokemon
- illustrationGAN: a famous GAN structure that used to generate anime face, project link is [here](https://github.com/tdrussell/IllustrationGAN)
- animeGAN: a GAN that used to generate high resolution anime gan, we found it in this [paper](https://arxiv.org/pdf/1708.05509.pdf), this GAN is based on DRFGAN, ResNet

## Implementation

pytorch, PIL, numpy