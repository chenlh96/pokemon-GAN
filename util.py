import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision


def save_checkpoint(epoch, generator, discriminator, filepath):
    assert filepath != None and 'pt' in filepath 
    len_ext = 2
    if 'pth' in filepath:
        len_ext = 3
    ext = filepath[-(len_ext + 1):]
    filename = filepath[: - (len_ext + 1)] + '_epoch_%d' % epoch
    filename = filename + ext
    torch.save({'generator': generator.state_dict(), 'discriminator': discriminator.state_dict()}, filename)

def load_checkpoint(epoch, generator, discriminator, path, device):
    assert path != None
    checkpoint = torch.load(path, map_location=device)
    generator.load_state_dict(checkpoint['generator'])  
    discriminator.load_state_dict(checkpoint['discriminator'])
    return generator, discriminator

def imshow(img_tensor):
    img = img_tensor.numpy()
    img = np.transpose(img, [1, 2, 0])
    if (img.shape[2] == 1):
        img = np.squeeze(img, axis = 2)
    plt.figure()
    plt.imshow(img)
    plt.show()

def make_figure_grid_dataset(dataset, grid_size):
    img_torch_grid = torch.zeros(grid_size ** 2, dataset[0][0].size(0), dataset[0][0].size(1), dataset[0][0].size(2))
    for i in range(grid_size ** 2):
        img_torch_grid[i] = dataset[i][0]
    return make_figure_grid(img_torch_grid, grid_size)

def make_figure_grid(img, grid_size, vmin=None, vmax=None, bright=0):
    assert type(img) == torch.Tensor or type(img) == np.ndarray
    if type(img) == np.ndarray:
        nc = np.argmin(img.shape[1:])
        if nc == 3:
            img = np.transpose(img, (0, nc, 1, 2))
        img = torch.from_numpy(img)
    assert img.size(0) <= grid_size ** 2
    grid_img = torchvision.utils.make_grid((img + 1) / 2 + bright, nrow=grid_size).detach().cpu()
    grid_img_np = np.transpose(grid_img.numpy(), (1, 2, 0))
    return grid_img_np

def make_gird_gif(img_list, grid_size, interval=50, delay=100):
    grid_img_list = []

    fig = plt.figure()
    for img in img_list:
        g_img = make_figure_grid(img, grid_size)
        im = plt.imshow(g_img, animated=True)
        grid_img_list.append([im])
    
    ani = animation.ArtistAnimation(fig, grid_img_list, interval=interval, blit=True, repeat_delay=delay)
    return ani


