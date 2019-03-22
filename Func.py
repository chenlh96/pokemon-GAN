import torch
import torch.nn as nn

def gen_noise(batch_size, dim):
    return torch.randn(batch_size, dim, 1, 1)

def show_weights(m):
    # if type(m) == nn.ConvTranspose2d:
    #     print(m.weights.data)
    if type(m) == nn.Conv2d:
        print(m.weights.data)

def save_checkpoint(epoch, generator, discriminator, path):
    pass

def load_checkpoint(epoch, generator, discriminator, path):
    pass
