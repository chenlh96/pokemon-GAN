import torch
import torch.nn as nn
import Func
from torch.utils.data import DataLoader

class generator(nn.Module):

    def __init__(self, dim_noise=100, dim_output_img=64, n_channel=3):
        super(generator, self).__init__()

        inplace = True
        init_kernel_sise = int(dim_output_img / (2 ** 4))
        
        self.conv_trans_2d1 = nn.ConvTranspose2d(dim_noise, dim_output_img*8, init_kernel_sise, 1, 0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(dim_output_img*8)
        self.relu1 = nn.ReLU(inplace=inplace)

        self.conv_trans_2d2 = nn.ConvTranspose2d(dim_output_img*8, dim_output_img*4, 4, 2,1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(dim_output_img*4)
        self.relu2 = nn.ReLU(inplace=inplace)

        self.conv_trans_2d3 = nn.ConvTranspose2d(dim_output_img*4, dim_output_img*2,  4,2,1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(dim_output_img*2)
        self.relu3 = nn.ReLU(inplace=inplace)

        self.conv_trans_2d4 = nn.ConvTranspose2d(dim_output_img*2, dim_output_img, 4,2,1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(dim_output_img)
        self.relu4 = nn.ReLU(inplace=inplace)

        # self.conv_trans_2d5 = nn.ConvTranspose2d(dim_output_img, dim_output_img, 4,2,1, bias=False)
        # self.batchnorm5 = nn.BatchNorm2d(dim_output_img)
        # self.relu5 = nn.ReLU(inplace=inplace)

        self.conv_trans_2d6 = nn.ConvTranspose2d(dim_output_img, n_channel, 4,2,1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.batchnorm1(self.conv_trans_2d1(x)))
        x = self.relu2(self.batchnorm2(self.conv_trans_2d2(x)))
        x = self.relu3(self.batchnorm3(self.conv_trans_2d3(x)))
        x = self.relu4(self.batchnorm4(self.conv_trans_2d4(x)))
        # x = self.relu5(self.batchnorm5(self.conv_trans_2d5(x)))
        x = self.tanh(self.conv_trans_2d6(x))
        return x


class discriminator(nn.Module):

    def __init__(self, dim_input_img=64, n_channel = 3):
        super(discriminator, self).__init__()

        slope = 0.2
        inplace = True
        final_ker_size = int(dim_input_img / (2**4))
        
        self.conv1 = nn.Conv2d(n_channel, dim_input_img, 4, 2,1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(dim_input_img)
        self.lrelu1 = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

        self.conv2 = nn.Conv2d(dim_input_img, dim_input_img * 2, 4, 2,1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(dim_input_img * 2)
        self.lrelu2 = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

        self.conv3 = nn.Conv2d(dim_input_img * 2, dim_input_img * 4, 4, 2,1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(dim_input_img * 4)
        self.lrelu3 = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

        # self.conv4 = nn.Conv2d(dim_input_img * 4, dim_input_img * 4, 4, 2,1, bias=False)
        # self.batchnorm4 = nn.BatchNorm2d(dim_input_img * 4)
        # self.lrelu4 = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

        self.conv5 = nn.Conv2d(dim_input_img * 4, dim_input_img * 8, 4, 2,1, bias=False)
        self.batchnorm5 = nn.BatchNorm2d(dim_input_img * 8)
        self.lrelu5 = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

        self.conv6 = nn.Conv2d(dim_input_img * 8, 1, final_ker_size, 1, 0, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.lrelu1(self.batchnorm1(self.conv1(x)))
        x = self.lrelu2(self.batchnorm2(self.conv2(x)))
        x = self.lrelu3(self.batchnorm3(self.conv3(x)))
        # x = self.lrelu4(self.batchnorm4(self.conv4(x)))
        x = self.lrelu5(self.batchnorm5(self.conv5(x)))
        x = self.sig(self.conv6(x))
        return x

# todo: 
# 3. enable the function to put the tensor and the model to a fix device, like GPU ***

def init_weight(layer):
    std = 0.02
    if type(layer) == nn.ConvTranspose2d:
        nn.init.normal_(layer.weight.data, mean=0, std=std)
    elif type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight.data, mean=0, std=std)
    elif type(layer) == nn.BatchNorm2d:
        nn.init.normal_(layer.weight.data, mean=1, std=std)
        nn.init.constant_(layer.bias.data, 0)

def train_base(epochs, batch_size, dim_noise, device, dataset, generator, discriminator, loss, optimizer_gen, optimizer_dis, filepath=None):
    # load the data
    worker = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    
    # create the list to store each loss
    loss_list, score_list, img_list = [], [], []
    num_fixed_ns_img = 64
    fixed_noise = torch.randn(num_fixed_ns_img, dim_noise, 1, 1, device=device)

    # start iterating the epoch
    for e in range(epochs):
        loss_dis, loss_gen, score_dis_real, score_dis_fake, score_gen = 0, 0, 0, 0, 0

        for i, data in enumerate(dataloader):
            b_size = batch_size
            if len(data[0]) < batch_size:
                b_size = len(data[0])
            # ---------------------------
            # 1. Train the discriminator
            # ---------------------------
            # generate noise samples from the generator
            batch_noise = torch.randn(b_size, dim_noise, 1, 1, device=device)
            fake_data = generator(batch_noise)

            # start to train the discriminator
            discriminator.zero_grad()
            # calculate the loss of the noise samples, which assigns the same label 0
            # for all the samples, and get the single output(marks) from the discriminator
            output = discriminator(fake_data.detach()).view(-1) # use .detach() to stop the requirement of gradient
            label = torch.full((b_size,), 0, device=device)
            loss_d_ns = loss(output, label)
            loss_d_ns.backward()
            score_dis_fake = output.mean().item()
            
            # calculate the loss of the real samples and assigns label 1 to represent
            # all samples are true and get the single output(marks) from the discriminator
            read_data = data[0].to(device)
            output = discriminator(read_data).view(-1)
            label.fill_(1)
            loss_d_real = loss(output, label)
            loss_d_real.backward()
            score_dis_real = output.mean().item()

            loss_d = loss_d_ns + loss_d_real
            loss_dis = loss_d.item()
            optimizer_dis.step()

            # ---------------------------
            # 2. Train the generator
            # ---------------------------
            # Feed the noise samplea to the discriminator agian to geit the accurate scores
            # after training the discriminator, and assign label 1 not to see the noise as
            # real label but to let the loss function to be correct and do correct back propogation
            generator.zero_grad()            
            # batch_noise = Func.torch.randn(b_size, dim_noise)
            # fake_data = generator(batch_noise)
            output = discriminator(fake_data).view(-1)
            loss_g = loss(output, label)
            loss_g.backward()
            score_gen = output.mean().item()
            loss_gen = loss_g.item()
            optimizer_gen.step()


            # print information to the console
            # print information 5 times in a epoch
            num2print = 30
            if (i + 1) % num2print == 0:
                print('epoch: %d, iter: %d, loss_D: %.4f, loss_G: %.4f;\t Scores: train D: D(x): %.4f, D(G(z)): %.4f train G: D(G(z))： %.4f'
                        % (e, (i + 1), loss_dis, loss_gen, score_dis_real, score_dis_fake, score_gen))           
                
                # store the final loss for D and G for a specific time interval of a whole epoch
                loss_list.append([loss_dis, loss_gen])
                # store the final score from D for noise and real samples for a specific time imterval on current epoch
                score_list.append([score_dis_fake, score_dis_real, score_gen])

        loss_list.append([loss_dis, loss_gen])
        score_list.append([score_dis_fake, score_dis_real, score_gen])
        # store the image that the generator create for each epoch
        test_img = generator(fixed_noise).detach().cpu()
        img_list.append(test_img.numpy())

        # save the model
        if (e + 1) % 5 == 0:
            Func.save_checkpoint(e, generator, discriminator, filepath)
    
    loss_list = list(map(list, zip(*loss_list)))
    score_list = list(map(list, zip(*score_list)))
        
    return generator, discriminator, loss_list, score_list, img_list

