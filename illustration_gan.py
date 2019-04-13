import torch
import torch.nn as nn
import util
import torch.optim as optim
from custom_layers import bilinear_upsample_deconv2d, minibatch_discrimination
from torch.utils.data import DataLoader

class auxiliary_fc_net(nn.Module):
    def __init__(self, dim_noise, dim_output_img, num_reduce_half, num_filter):
        super(auxiliary_fc_net, self).__init__()

        fc_size = 1024
        dim_feature_map = dim_output_img / (2 ** num_reduce_half)
        self.dim_imput = int((dim_feature_map ** 2) * num_filter)
        self.fc1 = nn.Linear(self.dim_imput, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.fc3 = nn.Linear(fc_size, fc_size)
        self.fc4 = nn.Linear(fc_size, dim_noise)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = x.view(-1, self.dim_imput)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

class generator_fc(nn.Module):
    def __init__(self, dim_noise, dim_output_img, num_reduce_half, num_filter):
        super(generator_fc, self).__init__()

        fc_size = 1024
        dim_feature_map = int(dim_output_img / (2 ** num_reduce_half))
        self.reshape_params = [-1, num_filter, dim_feature_map, dim_feature_map]
        self.fc1 = nn.Linear(dim_noise, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.fc3 = nn.Linear(fc_size, fc_size)
        self.fc4 = nn.Linear(fc_size, int((dim_feature_map ** 2) * num_filter))

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = x.view(self.reshape_params)
        return x

class generator_convt(nn.Module):

    def __init__(self, input_filters, dim_output_img=64, n_channel=3):
        super(generator_convt, self).__init__()

        inplace = True
        # init_kernel_sise = int(dim_output_img / (2 ** 4))
        
        # self.bilinear1 = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.conv1 = nn.Conv2d(input_filters, dim_output_img * 8, 5, 1, 2)
        self.bilinear_deconv1 = bilinear_upsample_deconv2d(2, input_filters, dim_output_img * 8, 5, 1, 2)
        self.batchnorm1 = nn.BatchNorm2d(dim_output_img*8)
        self.relu1 = nn.ReLU(inplace=inplace)

        self.bilinear_deconv2 = bilinear_upsample_deconv2d(2, dim_output_img * 8, dim_output_img * 4, 5, 1, 2)
        self.batchnorm2 = nn.BatchNorm2d(dim_output_img*4)
        self.relu2 = nn.ReLU(inplace=inplace)

        self.bilinear_deconv3 = bilinear_upsample_deconv2d(2, dim_output_img * 4, dim_output_img * 2, 5, 1, 2)
        self.batchnorm3 = nn.BatchNorm2d(dim_output_img*2)
        self.relu3 = nn.ReLU(inplace=inplace)

        self.bilinear_deconv4 = bilinear_upsample_deconv2d(2, dim_output_img * 2, dim_output_img, 5, 1, 2)
        self.batchnorm4 = nn.BatchNorm2d(dim_output_img)
        self.relu4 = nn.ReLU(inplace=inplace)

        self.conv = nn.Conv2d(dim_output_img, n_channel, 5, 1, 2, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.batchnorm1(self.bilinear_deconv1(x))
        x = self.relu1(x)
        x = self.batchnorm2(self.bilinear_deconv2(x))
        x = self.relu2(x)
        x = self.batchnorm3(self.bilinear_deconv3(x))
        x = self.relu3(x)
        x = self.batchnorm4(self.bilinear_deconv4(x))
        x = self.relu4(x)
        x = self.tanh(self.conv(x))
        return x

class generator(nn.Module):

    def __init__(self, dim_noise=100, dim_output_img=64, n_channel=3):
        super(generator, self).__init__()
        num_filter = dim_output_img * 16
        num_reduce_half = 4
        self.fc = generator_fc(dim_noise, dim_output_img, num_reduce_half, num_filter)
        self.convt = generator_convt(num_filter, dim_output_img, n_channel)
        self.auxiliary = auxiliary_fc_net(dim_noise, dim_output_img, num_reduce_half, num_filter)

    def forward(self, x):
        x_fc = self.fc(x)
        x_data = self.convt(x_fc)
        x_id = self.auxiliary(x_fc)

        return x_data, x_id


class discriminator(nn.Module):

    def __init__(self, dim_input_img=64, n_channel = 3):
        super(discriminator, self).__init__()

        slope = 0.2
        inplace = True
        proba = 0.5
        
        self.conv1 = nn.Conv2d(n_channel, dim_input_img, 5, 1, 2, bias=False)
        self.lrelu1 = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.do2 = nn.Dropout2d(p=proba, inplace=inplace)
        self.conv2 = nn.Conv2d(dim_input_img, dim_input_img * 2, 5, 1, 2, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(int(dim_input_img * 2 / 2)) # maxpool need to /2
        self.lrelu2 = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        self.do3 = nn.Dropout2d(p=proba, inplace=inplace)
        self.conv3 = nn.Conv2d(dim_input_img * 2, dim_input_img * 4, 5, 1, 2, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(int(dim_input_img * 4 / 2))
        self.lrelu3 = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
 
        self.do4 = nn.Dropout2d(p=proba, inplace=inplace)
        self.conv4 = nn.Conv2d(dim_input_img * 4, dim_input_img * 8, 5, 1, 2, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(int(dim_input_img * 8 / 2))
        self.lrelu4 = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        dim_output_feature = 100
        dim_c = 10
        dim_feature_map = int(dim_input_img / (2 ** 4))
        # assert dim_feature_map == 4

        self.flatten_size = dim_input_img * 8 * (dim_feature_map ** 2)
        self.miniDis = minibatch_discrimination(self.flatten_size, dim_output_feature, dim_c)

        fc_size = 1024

        self.fc1 = nn.Linear(self.flatten_size, fc_size)
        self.lrelu_fc1 = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.lrelu_fc2 = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
        self.fc3 = nn.Linear(fc_size, fc_size)
        self.lrelu_fc3 = nn.LeakyReLU(negative_slope=slope, inplace=inplace)

        self.fc4 = nn.Linear(fc_size + dim_output_feature, 1)
        # self.fc4 = nn.Linear(fc_size + self.flatten_size + dim_output_feature, 1)
        # self.lrelu_fc4 = nn.LeakyReLU(negative_slope=slope)
 

    def forward(self, x):
        x = self.maxpool1(self.lrelu1(self.conv1(x)))
        x = self.conv2(self.batchnorm2(self.do2(x)))
        x = self.maxpool2(self.lrelu2(x))
        x = self.conv3(self.batchnorm3(self.do3(x)))
        x = self.maxpool3(self.lrelu3(x))
        x = self.conv4(self.batchnorm4(self.do4(x)))
        x = self.maxpool4(self.lrelu4(x))

        x = x.view(-1, self.flatten_size)
        x_mini_dis = self.miniDis(x)

        x = self.lrelu_fc1(self.fc1(x))
        x = self.lrelu_fc2(self.fc2(x))
        x = self.lrelu_fc3(self.fc3(x))

        x = torch.cat([x, x_mini_dis], 1)
        x = self.fc4(x)

        return x

def init_weight(layer):
    std = 0.02
    if type(layer) == nn.ConvTranspose2d:
        nn.init.normal_(layer.weight.data, mean=0, std=std)
    elif type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight.data, mean=0, std=std)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight.data, mean=0, std=std)
        nn.init.normal_(layer.bias.data, mean=0, std=std)
    elif type(layer) == minibatch_discrimination:
         nn.init.normal_(layer.weight.data, mean=0, std=std)
    elif type(layer) == nn.BatchNorm2d:
        nn.init.normal_(layer.weight.data, mean=1, std=std)
        nn.init.constant_(layer.bias.data, 0)


def train_base(epochs, batch_size, dim_noise, device, dataset, generator, discriminator, loss, loss_auxiliary, optimizer_gen, optimizer_dis, filepath=None):
    # load the data
    worker = 2
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=worker)
    
    # create the list to store each loss
    loss_list, score_list, img_list = [], [], []
    num_fixed_ns_img = 64
    fixed_noise = torch.randn(num_fixed_ns_img, dim_noise, device=device)

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
            batch_noise = torch.randn(b_size, dim_noise, device=device)
            fake_data, noise_id = generator(batch_noise)

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
            loss_main = loss(output, label)
            loss_aux = loss_auxiliary(noise_id, batch_noise)
            loss_g = loss_main + loss_aux
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
        test_img, _ = generator(fixed_noise)
        test_img = test_img.detach().cpu()
        img_list.append(test_img.numpy())

        # save the model
        if (e + 1) % 5 == 0:
            util.save_checkpoint(e, generator, discriminator, filepath)
    
    loss_list = list(map(list, zip(*loss_list)))
    score_list = list(map(list, zip(*score_list)))
        
    return generator, discriminator, loss_list, score_list, img_list

def build_gen_dis(config):
    net_gen = generator(config.DIM_NOISE, config.DIM_IMG, config.N_CHANNEL).to(config.DEVICE)
    net_dis = discriminator(config.DIM_IMG, config.N_CHANNEL).to(config.DEVICE)

    if config.INIT:
        net_gen.apply(init_weight)
        net_dis.apply(init_weight)
    else:
        ext = config.PATH_MODEL[-4]
        path_model = config.PATH_IMPORT_MODEL[:-4] + '_epoch_%d' + ext % config.IMPORT_IDX_EPOCH
        net_gen, net_dis = util.load_checkpoint(config.EPOCHS, net_gen, net_dis, path_model)

    return net_gen, net_dis

def train(dataset, net_gen, net_dis, config):

    # config = config.config_illustration_gan
    # net_gen = generator(config.DIM_NOISE, config.DIM_IMG).to(config.DEVICE)
    # net_dis = discriminator(config.DIM_IMG).to(config.DEVICE)

    loss_main = nn.BCEWithLogitsLoss()
    loss_aux = nn.MSELoss()

    optim_gen = optim.Adam(net_gen.parameters(), lr=config.LEARNING_RATE, betas=(config.MOMENTUM, 0.99))
    optim_dis = optim.Adam(net_dis.parameters(), lr=config.LEARNING_RATE, betas=(config.MOMENTUM, 0.99))

    net_gen, net_dis, losses, _, imgs = train_base(config.EPOCHS, config.BATCH_SIZE, config.DIM_NOISE, config.DEVICE,
                                                    dataset, net_gen, net_dis, loss_main, loss_aux, optim_gen, optim_dis, config.PATH_MODEL)
    
    return net_gen, net_dis, losses, imgs

