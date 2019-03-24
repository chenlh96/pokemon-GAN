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
        self.dim_input_img = dim_input_img
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

# todo: 
# 3. enable the function to put the tensor and the model to a fix device, like GPU ***

def init_weight(layer):
    if type(layer) == nn.ConvTranspose2d:
        nn.init.normal_(layer.weight.data, mean=0, std=0.2)
    elif type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight.data, mean=0, std=0.2)
    elif type(layer) == nn.BatchNorm2d:
        nn.init.normal_(layer.weight.data, mean=1, std=0.2)
        nn.init.constant_(layer.bias.data, 0)

def train_base(epochs, batch_size, dim_noise, dim_img, device, dataset, generator, discriminator, loss, optimizer_gen, optimizer_dis, filepath=None, filename='dcgan.pth'):
    # load the data
    worker = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    # create the list to store each loss
    loss_list, score_list, img_list = [], [], []
    num_fixed_ns_img = 6
    fixed_noise = Func.gen_noise(num_fixed_ns_img, dim_noise).to(device)

    # start iterating the epoch
    for e in range(epochs):
        # iterating the mini-bathc
        loss_d_epoch, loss_g_epoch = 0,0
        score_ns_epoch, score_real_epoch, score_dis_epoch = 0, 0, 0
        img_epoch = []

        for i, data in enumerate(dataloader):
            # ---------------------------
            # 1. Train the discriminator
            # ---------------------------
            # generate noise samples from the generator
            batch_noise = Func.gen_noise(batch_size, dim_noise).to(device)
            output_gen = generator(batch_noise)
            # check if the output size is legal
            assert output_gen.size() == torch.Size([batch_size, 3, dim_img, dim_img])

            # start to train the discriminator
            discriminator.zero_grad()
            # calculate the loss of the noise samples, which assigns the same label 0
            # for all the samples, and get the single output(marks) from the discriminator
            output_dis_ns = discriminator(output_gen.detach()).view(-1) # use .detach() to stop the gradient pass the generator
            ns_label = torch.zeros(batch_size).to(device)
            loss_d_real = loss(output_dis_ns, ns_label)
            loss_d_real.backward()
            optimizer_dis.step()
            
            # calculate the loss of the real samples and assigns label 1 to represent
            # all samples are true and get the single output(marks) from the discriminator
            read_data = data[0].to(device)
            output_dis_real = discriminator(read_data).view(-1)
            real_label = torch.ones(batch_size, device=device)
            loss_d_ns = loss(output_dis_real, real_label)
            loss_d_ns.backward()
            optimizer_dis.step()

            # ---------------------------
            # 2. Train the generator
            # ---------------------------
            # Feed the noise samplea to the discriminator agian to geit the accurate scores
            # after training the discriminator, and assign label 1 not to see the noise as
            # real label but to let the loss function to be correct and do correct back propogation
            generator.zero_grad()            
            # batch_noise = Func.gen_noise(batch_size, dim_noise)
            # output_gen = generator(batch_noise)
            output_dis = discriminator(output_gen).view(-1)
            ns_label = torch.ones(batch_size, device=device)
            loss_g = loss(output_dis, ns_label)
            loss_g.backward()
            optimizer_gen.step()

            # calculate the loss and store it
            loss_d_epoch = loss_d_ns.item() + loss_d_real.item()
            loss_g_epoch = loss_g.item()
            # store the final score of the current epoch using the trained D of last epoch
            score_ns_epoch = output_dis_ns.mean().item()
            score_real_epoch = output_dis_real.mean().item()
            score_dis_epoch = output_dis.mean().item()


            # print information to the console
            if (i + 1) % 2 == 0:
                print('epoch: %d, iter: %d, loss_D: %.4f, loss_G: %.4f'
                        % (e, (i + 1), loss_d_epoch, loss_g_epoch))
                print('train D: D(x): %.4f, D(G(z)): %.4f train G: D(G(z))： %.4f'
                        % (score_real_epoch, score_ns_epoch, score_dis_epoch))            
                
                # store the final loss for D and G for a specific time interval of a whole epoch
                loss_list.append([loss_d_epoch, loss_g_epoch])
                # store the final score from D for noise and real samples for a specific time imterval on current epoch
                score_list.append([score_ns_epoch, score_real_epoch, score_dis_epoch])
                # store the image that the generator create
                img_epoch = []
                test_img = generator(fixed_noise).detach()
                img_epoch.append(test_img.cpu().numpy())
                img_list.append(img_epoch)
        
        Func.save_checkpoint(e, generator, discriminator, filepath, filename)
        
    return generator, discriminator, loss_list, score_list, img_list

