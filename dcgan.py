import torch
import torch.nn as nn
import Func
from torch.utils.data import DataLoader

class generator(nn.Module):

    def __init__(self, dim_noise=100, dim_output_img=64):
        super(generator, self).__init__()
        self.conv_trans_2d1 = nn.ConvTranspose2d(dim_noise, dim_output_img*16, 4, 1, 0)
        self.batchnorm1 = nn.BatchNorm2d(1024)
        self.relu1 = nn.ReLU()

        self.conv_trans_2d2 = nn.ConvTranspose2d(dim_output_img*16, dim_output_img*8, 4, 2,1)
        self.batchnorm2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU()

        self.conv_trans_2d3 = nn.ConvTranspose2d(dim_output_img*8, dim_output_img*4,  4,2,1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.conv_trans_2d4 = nn.ConvTranspose2d(dim_output_img*4, dim_output_img*2, 4,2,1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        self.conv_trans_2d5 = nn.ConvTranspose2d(dim_output_img*2, 3, 4,2,1)
        self.batchnorm5 = nn.BatchNorm2d(3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.batchnorm1(self.conv_trans_2d1(x)))
        x = self.relu2(self.batchnorm2(self.conv_trans_2d2(x)))
        x = self.relu3(self.batchnorm3(self.conv_trans_2d3(x)))
        x = self.relu4(self.batchnorm4(self.conv_trans_2d4(x)))
        x = self.tanh(self.batchnorm5(self.conv_trans_2d5(x)))
        return x


class discriminator(nn.Module):

    def __init__(self, dim_input_img=64):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, dim_input_img * 2, 5, 2, 2)
        self.batchnorm1 = nn.BatchNorm2d(dim_input_img * 8)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv2 = nn.Conv2d(dim_input_img * 2, dim_input_img * 4, 5, 2, 2)
        self.batchnorm2 = nn.BatchNorm2d(dim_input_img * 4)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2)

        self.conv3 = nn.Conv2d(dim_input_img * 4, dim_input_img * 8, 3, 2, 1)
        self.batchnorm3 = nn.BatchNorm2d(dim_input_img * 8)
        self.lrelu3 = nn.LeakyReLU(negative_slope=0.2)

        self.conv4 = nn.Conv2d(dim_input_img * 8, dim_input_img * 8, 3, 2, 1)
        self.batchnorm4 = nn.BatchNorm2d(dim_input_img * 8)
        self.lrelu4 = nn.LeakyReLU(negative_slope=0.2)

        self.conv5 = nn.Conv2d(dim_input_img * 8, 1, 4, 1, 0)
        self.batchnorm5 = nn.BatchNorm2d(1)
        self.lrelu5 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.lrelu1(self.batchnorm1(self.conv1(x)))
        x = self.lrelu2(self.batchnorm2(self.conv2(x)))
        x = self.lrelu3(self.batchnorm3(self.conv3(x)))
        x = self.lrelu4(self.batchnorm4(self.conv4(x)))
        x = self.lrelu5(self.batchnorm5(self.conv5(x)))

def train(epochs, batch_size, dim_noise, dataset, generator, discriminator, loss, optimizer_gen, optimizer_dis):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for e in range(epochs):
        for i, data in enumerate(dataloader):
            batch_noise = Func.gen_noise(batch_size, dim_noise)
            output_gen = generator(batch_noise)

            discriminator.zero_grad()

            output_dis_ns = discriminator(output_gen.detach()).view(-1)
            output_dis_real = discriminator(data).view(-1)

            real_label = torch.ones(batch_size)
            loss_d = loss(output_dis_real, real_label)
            loss_d.backword()
            ns_label = torch.zeros(batch_size)
            loss_d = loss(output_dis_ns, ns_label)
            loss_d.backword()
            optimizer_dis.step()

            batch_noise = Func.gen_noise(batch_size, dim_noise)
            output_gen = generator(batch_noise)
            output_dis = discriminator(output_gen.detach()).view(-1)

            generator.zero_grad()
            ns_label = torch.zeros(batch_size)
            loss_g = loss(output_dis, ns_label)
            loss_g.backword()
            optimizer_gen.step()


