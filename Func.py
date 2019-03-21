import torch
import torch.nn as nn

def init_weight(layer):
    if type(layer) == nn.ConvTranspose2d:
        nn.init.normal_(layer.weight.data, mean=0, std=0.2)
    elif type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight.data, mean=0, std=0.2)
    elif type(layer) == nn.BatchNorm2d:
        nn.init.normal_(layer.weight.data, mean=1, std=0.2)
        nn.init.constant_(layer.bias.data, 0)

def gen_noise(batch_size, dim):
    return torch.randn(batch_size, dim, 1, 1)

def show_weights(m):
    # if type(m) == nn.ConvTranspose2d:
    #     print(m.weights.data)
    if type(m) == nn.Conv2d:
        print(m.weights.data)
