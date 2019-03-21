import torch
import torch.nn as nn
import Func
from torch.utils.data import DataLoader

class generator(nn.Module):

    def __init__(self, dim_noise=100, dim_output_img=64):
        super(generator, self).__init__()
        self.dim_noise = dim_noise
        self.dim_output_img = dim_output_img
        self.conv_trans_2d1 = nn.ConvTranspose2d(dim_noise, dim_output_img*16, 4, 1, 0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(dim_output_img*16)
        self.relu1 = nn.ReLU()

        self.conv_trans_2d2 = nn.ConvTranspose2d(dim_output_img*16, dim_output_img*8, 4, 2,1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(dim_output_img*8)
        self.relu2 = nn.ReLU()

        self.conv_trans_2d3 = nn.ConvTranspose2d(dim_output_img*8, dim_output_img*4,  4,2,1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(dim_output_img*4)
        self.relu3 = nn.ReLU()

        self.conv_trans_2d4 = nn.ConvTranspose2d(dim_output_img*4, dim_output_img*2, 4,2,1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(dim_output_img*2)
        self.relu4 = nn.ReLU()

        self.conv_trans_2d5 = nn.ConvTranspose2d(dim_output_img*2, dim_output_img*2, 4,2,1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(dim_output_img*2)
        self.relu5 = nn.ReLU()

        self.conv_trans_2d6 = nn.ConvTranspose2d(dim_output_img*2, 3, 4,2,1, bias=False)
        self.batchnorm6 = nn.BatchNorm2d(3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.batchnorm1(self.conv_trans_2d1(x)))
        x = self.relu2(self.batchnorm2(self.conv_trans_2d2(x)))
        x = self.relu3(self.batchnorm3(self.conv_trans_2d3(x)))
        x = self.relu4(self.batchnorm4(self.conv_trans_2d4(x)))
        x = self.relu5(self.batchnorm5(self.conv_trans_2d5(x)))
        x = self.tanh(self.batchnorm6(self.conv_trans_2d6(x)))
        return x


class discriminator(nn.Module):

    def __init__(self, dim_input_img=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, dim_input_img * 2, 5, 2, 2, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(dim_input_img * 2)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv2 = nn.Conv2d(dim_input_img * 2, dim_input_img * 4, 5, 2, 2, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(dim_input_img * 4)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2)

        self.conv3 = nn.Conv2d(dim_input_img * 4, dim_input_img * 8, 3, 2, 1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(dim_input_img * 8)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.2)

        self.conv4 = nn.Conv2d(dim_input_img * 8, dim_input_img * 8, 3, 2, 1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(dim_input_img * 8)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.2)

        self.conv5 = nn.Conv2d(dim_input_img * 8, dim_input_img * 16, 3, 2, 1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(dim_input_img * 16)
        self.lrelu5 = nn.LeakyReLU(negative_slope=0.2)

        self.conv6 = nn.Conv2d(dim_input_img * 16, 1, 4, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.lrelu1(self.batchnorm1(self.conv1(x)))
        x = self.lrelu2(self.batchnorm2(self.conv2(x)))
        x = self.lrelu3(self.batchnorm3(self.conv3(x)))
        x = self.lrelu4(self.batchnorm4(self.conv4(x)))
        x = self.lrelu5(self.batchnorm5(self.conv5(x)))
        x = self.sig(self.conv6(x))
        return x

def train_base(epochs, batch_size, dim_noise, dim_img, dataset, generator, discriminator, loss, optimizer_gen, optimizer_dis):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_gen, loss_dic = 0, 0


    for e in range(epochs):
        for i, data in enumerate(dataloader):
            batch_noise = Func.gen_noise(batch_size, dim_noise)
            output_gen = generator(batch_noise)

            assert output_gen.size() == torch.Size([batch_size, 3, dim_img, dim_img])

            discriminator.zero_grad()
            output_dis_ns = discriminator(output_gen.detach()).view(-1)
            output_dis_real = discriminator(data[0]).view(-1)

            real_label = torch.ones(batch_size)
            loss_d = loss(output_dis_real, real_label)
            loss_d.backward()
            
            ns_label = torch.zeros(batch_size)
            loss_d = loss(output_dis_ns, ns_label)
            loss_d.backward()
            optimizer_dis.step()

            batch_noise = Func.gen_noise(batch_size, dim_noise)
            output_gen = generator(batch_noise)
            output_dis = discriminator(output_gen).view(-1)

            # todo: find out the problem that the params of gen cannot update. The reason might be:
            # 1. the grad cannot pass to param of gen;
            # 2. the loss becomes 0

            generator.zero_grad()
            ns_label = torch.ones(batch_size)
            loss_g = loss(output_dis, ns_label)
            print(loss_g)
            loss_g.backward()
            optimizer_gen.step()

            break

    return generator, discriminator, loss_gen, loss_dic

