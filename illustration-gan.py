import torch
import torch.nn as nn
import Func
from torch.utils.data import DataLoader

class generator(nn.Module):

    def __init__(self, dim_noise=100, dim_output_img=64):
        super(generator, self).__init__()
        self.dim_noise = dim_noise
        self.inplace=True
        self.dim_output_img = dim_output_img
        self.conv_trans_2d1 = nn.ConvTranspose2d(dim_noise, dim_output_img*16, 4, 1, 0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(dim_output_img*16)
        self.relu1 = nn.ReLU(inplace=self.inplace)

        self.conv_trans_2d2 = nn.ConvTranspose2d(dim_output_img*16, dim_output_img*8, 4, 2,1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(dim_output_img*8)
        self.relu2 = nn.ReLU(inplace=self.inplace)

        self.conv_trans_2d3 = nn.ConvTranspose2d(dim_output_img*8, dim_output_img*4,  4,2,1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(dim_output_img*4)
        self.relu3 = nn.ReLU(inplace=self.inplace)

        self.conv_trans_2d4 = nn.ConvTranspose2d(dim_output_img*4, dim_output_img*2, 4,2,1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(dim_output_img*2)
        self.relu4 = nn.ReLU(inplace=self.inplace)

        self.conv_trans_2d5 = nn.ConvTranspose2d(dim_output_img*2, dim_output_img*2, 4,2,1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(dim_output_img*2)
        self.relu5 = nn.ReLU(inplace=self.inplace)

        self.conv_trans_2d6 = nn.ConvTranspose2d(dim_output_img*2, 3, 4,2,1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.batchnorm1(self.conv_trans_2d1(x)))
        x = self.relu2(self.batchnorm2(self.conv_trans_2d2(x)))
        x = self.relu3(self.batchnorm3(self.conv_trans_2d3(x)))
        x = self.relu4(self.batchnorm4(self.conv_trans_2d4(x)))
        x = self.relu5(self.batchnorm5(self.conv_trans_2d5(x)))
        x = self.tanh(self.conv_trans_2d6(x))
        return x


class discriminator(nn.Module):

    def __init__(self, dim_input_img=64):
        super(discriminator, self).__init__()
        self.slope = 0.2
        self.inplace=True
        self.dim_input_img = dim_input_img
        self.conv1 = nn.Conv2d(3, dim_input_img * 2, 4, 2,1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(dim_input_img * 2)
        self.lrelu1 = nn.LeakyReLU(negative_slope=self.slope, inplace=self.inplace)

        self.conv2 = nn.Conv2d(dim_input_img * 2, dim_input_img * 4, 4, 2,1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(dim_input_img * 4)
        self.lrelu2 = nn.LeakyReLU(negative_slope=self.slope, inplace=self.inplace)

        self.conv3 = nn.Conv2d(dim_input_img * 4, dim_input_img * 8, 4, 2,1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(dim_input_img * 8)
        self.lrelu3 = nn.LeakyReLU(negative_slope=self.slope, inplace=self.inplace)

        self.conv4 = nn.Conv2d(dim_input_img * 8, dim_input_img * 8, 4, 2,1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(dim_input_img * 8)
        self.lrelu4 = nn.LeakyReLU(negative_slope=self.slope, inplace=self.inplace)

        self.conv5 = nn.Conv2d(dim_input_img * 8, dim_input_img * 16, 4, 2,1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(dim_input_img * 16)
        self.lrelu5 = nn.LeakyReLU(negative_slope=self.slope, inplace=self.inplace)

        self.conv6 = nn.Conv2d(dim_input_img * 16, 1, 4, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.lrelu1(self.batchnorm1(self.conv1(x)))
        x = self.lrelu2(self.batchnorm2(self.conv2(x)))
        x = self.lrelu3(self.batchnorm3(self.conv3(x)))
        x = self.lrelu4(self.batchnorm4(self.conv4(x)))
        x = self.lrelu5(self.batchnorm5(self.conv5(x)))
        x = self.sig(self.conv6(x))
        return x