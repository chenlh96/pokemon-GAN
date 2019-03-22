import torch
import torch.nn as nn

def gen_noise(batch_size, dim):
    return torch.randn(batch_size, dim, 1, 1)

def show_weights(m):
    # if type(m) == nn.ConvTranspose2d:
    #     print(m.weights.data)
    if type(m) == nn.Conv2d:
        print(m.weights.data)

def save_checkpoint(epoch, generator, discriminator, path, filename):
    if filename == None or path == None:
        return
    len_ext = 2
    ext = '.pt'
    if 'pth' in filename:
        len_ext = 3
        ext = '.pth'
    filename = filename[:-(len_ext+1)] + '_%d_' % epoch + ext
    file_path = '{}/{}'.format(path, filename)
    torch.save({'generator': generator.state_dict(), 'discriminator': discriminator.state_dict()}, file_path)

def load_checkpoint(epoch, generator, discriminator, path, filename):
    if filename == None or path == None:
        return
    file_path = '{}/{}'.format(path, filename)
    checkpoint = torch.load(file_path)
    generator.load_state_dict(checkpoint['generator'])  
    discriminator.load_state_dict(checkpoint['discriminator'])

