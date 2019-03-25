import torch
import torch.nn as nn


def save_checkpoint(epoch, generator, discriminator, path, filename):
    assert filename != None and path != None
    assert 'pt' in filename
    len_ext = 2
    ext = '.pt'
    if 'pth' in filename:
        len_ext = 3
        ext = '.pth'
    filename = filename[:-(len_ext+1)] + '_%d' % epoch + ext
    file_path = '{}/{}'.format(path, filename)
    torch.save({'generator': generator.state_dict(), 'discriminator': discriminator.state_dict()}, file_path)

def load_checkpoint(epoch, generator, discriminator, path, filename):
    assert filename != None and path != None
    file_path = '{}/{}'.format(path, filename)
    checkpoint = torch.load(file_path)
    generator.load_state_dict(checkpoint['generator'])  
    discriminator.load_state_dict(checkpoint['discriminator'])

def make_figure_grid_fr_tensor(img_tensor, grid_size):
    assert type(img_tensor) == torch.Tensor
    pass

