import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision


def save_checkpoint(epoch, generator, discriminator, filepath):
    assert filepath != None and 'pt' in filepath 
    len_ext = 2
    if 'pth' in filepath:
        len_ext = 3
    ext = filepath[-(len_ext+1)]
    filename = filepath[:-(len_ext+1)] + '_epoch_%d' % epoch + ext
    torch.save({'generator': generator.state_dict(), 'discriminator': discriminator.state_dict()}, filename)

def load_checkpoint(epoch, generator, discriminator, path):
    assert path != None
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint['generator'])  
    discriminator.load_state_dict(checkpoint['discriminator'])
    return generator, discriminator

def make_figure_grid(img_list, grid_size, vmin=None, vmax=None):
    assert type(img_list) == torch.Tensor or type(img_list) == np.ndarray
    if type(img_list) == np.ndarray:
        nc = np.argmin(img_list.shape[1:])
        if nc == 3:
            img_list = np.transpose(img_list, (0, nc, 1, 2))
        img_list = torch.from_numpy(img_list)
        print(img_list.size(0))
    assert img_list.size(0) <= grid_size ** 2
    grid_img = torchvision.utils.make_grid(img_list, nrow=grid_size).detach().cpu()
    grid_img_np = np.transpose(grid_img.numpy(), (1, 2, 0))
    return grid_img_np


