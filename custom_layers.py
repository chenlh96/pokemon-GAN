import torch
import torch.nn as nn


class bn_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(bn_conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class bn_transpose_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(bn_transpose_conv2d, self).__init__()
        self.conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv_t(x))


class bilinear_deconv2d(nn.Module):
    def __init__(self, scale_factor, n_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(bilinear_deconv2d, self).__init__()
        self.bilinear = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.conv = nn.Conv2d(n_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True)

    def forward(self, x):
        x = self.bilinear(x)
        x = self.conv(x)
        return x


class bn_bilinear_deconv2d(nn.Module):
    def __init__(self, scale_factor, n_channels, out_channels, kernel_size, stride=1, padding=0):
        super(bn_bilinear_deconv2d, self).__init__()
        self.bilinear = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.conv = nn.Conv2d(n_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(self.bilinear(x))
        x = self.bn(x)
        return x


class minibatch_discrimination(nn.Module):
    def __init__(self, dim_input_feature, dim_output_feature, c):
        super(minibatch_discrimination, self).__init__()
        self.input_feture = dim_input_feature
        self.output_feature = dim_input_feature
        self.output_feature = dim_output_feature
        self.c = c
        self.weight = nn.Parameter(torch.empty(self.input_feture, self.output_feature * self.c))
        self.bias = nn.Parameter(torch.empty(self.output_feature))
        
    def forward(self, x):
        x = x.view(-1, self.input_feture)
        mat = torch.mm(x, self.weight)
        mat = mat.view(-1, self.output_feature, self.c)

        mat = mat.unsqueeze(0)
        mat_T = mat.permute(1, 0, 2, 3)
        output = torch.exp(-torch.abs(mat - mat_T).sum(3))
        output = output.sum(0) + self.bias

        # output = torch.cat([x, output], 1)
        return output


class sr_resBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels=64, kernel_size=3, stride=1, padding=1):
        super(sr_resBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_copy = x
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + x_copy
        return x


class sub_pixel_deconv2d(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels=256, kernel_size=3, stride=1, padding=1):
        super(sub_pixel_deconv2d, self).__init__()
        out_channels = (scale_factor ** 2) *in_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pshuffle = nn.PixelShuffle(scale_factor)
        self.bn = nn.BatchNorm2d(int(out_channels / (scale_factor ** 2)))
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.pshuffle(self.conv(x))
        x = self.relu(self.bn(x))
        return x


class dis_resBlock(nn.Module):
    def __init__(self, in_channels=32, kernel_size=3, stride=1, padding=1, activate_before_addition=True):
        super(dis_resBlock, self).__init__()
        self.add_activate = activate_before_addition
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.lrelu1 = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding)
        self.lrelu2 = nn.LeakyReLU(0.2, True)
        self.lrelu3 = nn.LeakyReLU(0.2, True)

    
    def forward(self, x):
        x_id = x
        x = self.lrelu1(self.conv1(x))
        x = self.conv2(x)
        if self.add_activate:
            x = self.lrelu2(x)
        x = self.lrelu3(x + x_id)
        return x

