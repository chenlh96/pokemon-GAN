import torch
import torch.nn as nn

def init_weight(layer, std):
    if type(layer) == nn.ConvTranspose2d:
        nn.init.normal_(layer.weight.data, mean=0, std=std)
    elif type(layer) == nn.BatchNorm2d:
        nn.init.normal_(layer.weight.data, mean=1, std=std)
        nn.init.constant_(layer.bias.data, 0)

def gen_noise(batch_size, dim):
    return torch.randn(batch_size, dim)