import torch
import torch.nn as nn

class bilinear_upsample_deconv2d(nn.Module):
    def __init__(self, scale_factor, n_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(bilinear_upsample_deconv2d, self).__init__()

        self.bilinear = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.conv = nn.Conv2d(n_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True)

        # self.weight = nn.Parameter(torch.Tensor(self.conv.weight))
        # self.bias = nn.Parameter(self.conv.bias)


    def forward(self, x):
        x = self.bilinear(x)
        x = self.conv(x)
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

        output = torch.cat([x, output], 1)
        return output

