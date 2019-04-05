import torch
import torch.nn as nn

class bilinear_upsample_deconv2d(nn.Module):
    def __init__(self, scale_factor, n_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(bilinear_upsample_deconv2d, self).__init__()

        self.bilinear = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.conv = nn.Conv2d(n_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        x = self.bilinear(x)
        x = self.conv(x)
        return x

class minibatch_discrimination(nn.Module):
    def __init__(self):
        super(minibatch_discrimination, self).__init__()
        
    def forward(self, x):
         return x
