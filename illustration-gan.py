import torch
import torch.nn as nn
import Func
from custom_layers import bilinear_upsample_deconv2d
from torch.utils.data import DataLoader

class auxiliary_fc_net(nn.Module):
    def __init__(self, dim_noise, dim_output_img, num_reduce_half, num_filter):
        super(auxiliary_fc_net, self).__init__()

        fc_size = 1024
        dim_feature_map = dim_output_img / (2 ** num_reduce_half)
        self.fc1 = nn.Linear((dim_feature_map ** 2) * num_filter, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.fc3 = nn.Linear(fc_size, fc_size)
        self.fc4 = nn.Linear(fc_size, dim_noise)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = torch.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        return x

class generator_fc(nn.Module):
    def __init__(self, dim_noise, dim_output_img, num_reduce_half, num_filter):
        super(generator_fc, self).__init__()

        fc_size = 1024
        dim_feature_map = dim_output_img / (2 ** num_reduce_half)
        self.reshape_params = [num_filter, dim_feature_map, dim_feature_map]
        self.fc1 = nn.Linear(dim_noise, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.fc3 = nn.Linear(fc_size, fc_size)
        self.fc4 = nn.Linear(fc_size, (dim_feature_map ** 2) * num_filter)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = x.view(self.reshape_params)
        return x

class generator_convt(nn.Module):

    def __init__(self, input_feature_map, dim_output_img=64, n_channel=3):
        super(generator_convt, self).__init__()

        inplace = True
        # init_kernel_sise = int(dim_output_img / (2 ** 4))
        
        # self.bilinear1 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.conv1 = nn.Conv2d(input_feature_map, dim_output_img * 8, 5, 1, 2)
        self.bilinear_deconv1 = bilinear_upsample_deconv2d(2, input_feature_map, dim_output_img * 8, 5, 1, 2)
        self.batchnorm1 = nn.BatchNorm2d(dim_output_img*8)
        self.relu1 = nn.ReLU(inplace=inplace)

        self.bilinear_deconv2 = bilinear_upsample_deconv2d(2, dim_output_img * 8, dim_output_img * 4, 5, 1, 2)
        self.batchnorm2 = nn.BatchNorm2d(dim_output_img*4)
        self.relu2 = nn.ReLU(inplace=inplace)

        self.bilinear_deconv3 = bilinear_upsample_deconv2d(2, dim_output_img * 4, dim_output_img * 2, 5, 1, 2)
        self.batchnorm3 = nn.BatchNorm2d(dim_output_img*2)
        self.relu3 = nn.ReLU(inplace=inplace)

        self.bilinear_deconv4 = bilinear_upsample_deconv2d(2, dim_output_img * 2, dim_output_img, 5, 1, 2)
        self.batchnorm4 = nn.BatchNorm2d(dim_output_img)
        self.relu4 = nn.ReLU(inplace=inplace)


        self.conv = nn.Conv2d(dim_output_img, n_channel, 5, 1, 2, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.batchnorm1(self.bilinear_deconv1(x))
        x = self.relu1(x)
        x = self.batchnorm2(self.bilinear_deconv2(x))
        x = self.relu2(x)
        x = self.batchnorm3(self.bilinear_deconv3(x))
        x = self.relu3(x)
        x = self.batchnorm4(self.bilinear_deconv4(x))
        x = self.relu4(x)
        x = self.tanh(self.conv(x))
        return x


class discriminator(nn.Module):

    def __init__(self, dim_input_img=64, n_channel = 3):
        super(discriminator, self).__init__()

        slope = 0.2
        inplace = True
        proba = 0.5
        
        self.conv1 = nn.Conv2d(n_channel, dim_input_img, 5, 1, 2, bias=False)
        self.lrelu1 = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.do2 = nn.Dropout(p=proba, inplace=inplace)
        self.conv2 = nn.Conv2d(dim_input_img, dim_input_img * 2, 5, 1, 2, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(dim_input_img * 2)
        self.lrelu2 = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        self.do3 = nn.Dropout(p=proba, inplace=inplace)
        self.conv3 = nn.Conv2d(dim_input_img * 2, dim_input_img * 4, 5, 1, 2, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(dim_input_img * 4)
        self.lrelu3 = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        self.do4 = nn.Dropout(p=proba, inplace=inplace)
        self.conv4 = nn.Conv2d(dim_input_img * 4, dim_input_img * 8, 5, 1, 2, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(dim_input_img * 8)
        self.lrelu4 = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        fc_size = 1024

        self.fc1 = nn.Linear(dim_input_img * 8, fc_size)
        self.lrelu_fc1 = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.lrelu_fc2 = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
        self.fc3 = nn.Linear(fc_size, fc_size)
        self.lrelu_fc3 = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
 

    def forward(self, x):

        return x

def train_illustrate():
    pass