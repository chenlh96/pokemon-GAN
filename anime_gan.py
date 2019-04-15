import torch
import torch.nn as nn
import util
import torch.optim as optim
import custom_layers as op
from torch.utils.data import DataLoader


class generator(nn.Module):

    def __init__(self, dim_noise, dim_filter, num_filter, dim_output_img=64, n_channel=3):
        super(generator, self).__init__()

        inplace = True
        self.fc = nn.Linear(dim_noise, 64 * (16 ** 2))
        self.bn1 = nn.BatchNorm1d(64 * (16 ** 2))
        self.relu1 = nn.ReLU(inplace=inplace)
        self.block = nn.ModuleList([op.sr_resBlock(64) for _ in range(16)])
        self.bn2 = nn.BatchNorm1d(64 * (16 ** 2))
        self.relu2 = nn.ReLU(inplace=inplace)
        self.sub_pixel_cnn = nn.ModuleList([op.sub_pixel_cnn(2, 64)])
        self.sub_pixel_cnn.append([op.sub_pixel_cnn(2, 256) for _ in range(2)])
        self.conv = nn.Conv2d(256, n_channel, 9, 1, 0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc(x)))
        x_id = x
        for bk in self.block:
            x = bk(x)
        x = self.relu2(self.bn(x)) + x_id
        for sub in self.sub_pixel_cnn:
            x = sub(x)
        x = self.tanh(self.conv(x))
        return x


class discriminator(nn.Module):

    def __init__(self, dim_input_img=128, n_channel = 3):
        super(discriminator, self).__init__()

        slope = 0.2
        inplace=True
        self.basic1 = nn.ModuleList([nn.Conv2d(n_channel, 32, 4, 2, 1), nn.LeakyReLU(slope, inplace)])
        self.basic1.append([op.dis_resBlock(32) for _ in range(2)])
        self.basic2 = nn.ModuleList([nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(slope, inplace)])
        self.basic2.append([op.dis_resBlock(64) for _ in range(4)])
        self.basic3 = nn.ModuleList([nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(slope, inplace)])
        self.basic3.append([op.dis_resBlock(128) for _ in range(4)])
        self.basic4 = nn.ModuleList([nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(slope, inplace)])
        self.basic4.append([op.dis_resBlock(256) for _ in range(4)])
        self.basic5 = nn.ModuleList([nn.Conv2d(256, 512, 4, 2, 1), nn.LeakyReLU(slope, inplace)])
        self.basic5.append([op.dis_resBlock(512) for _ in range(4)])
        self.basic6 = nn.ModuleList([nn.Conv2d(512, 1024, 3, 2, 1), nn.LeakyReLU(slope, inplace)])
        self.block = nn.ModuleList([self.basic1, self.basic2, self.basic3, self.basic4, self.basic5, self.basic6])
        num_reduce_half = 5
        dim_final_kernel = int(dim_input_img / (2 ** num_reduce_half))
        flatten_size = 1024 * (dim_final_kernel ** 2)
        self.fc_score = nn.Linear(flatten_size, 1)
        self.fc_label = nn.Linear(flatten_size, 34)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        for bk in self.block:
            for bs in bk:
                x = bs(x)
        x_score = self.sig(self.fc_score(x))
        x_label = self.sig(self.fc_label(x))

        return x_score, x_label

def init_weight(layer):
    std = 0.02
    if type(layer) == nn.ConvTranspose2d:
        nn.init.normal_(layer.weight.data, mean=0, std=std)
    elif type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight.data, mean=0, std=std)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight.data, mean=0, std=std)
        nn.init.normal_(layer.bias.data, mean=0, std=std)
    elif type(layer) == op.minibatch_discrimination:
         nn.init.normal_(layer.weight.data, mean=0, std=std)
         nn.init.constant_(layer.bias.data, 0)
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
        test_img = generator(fixed_noise)
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

    optim_gen = optim.Adam(net_gen.parameters(), lr=config.LEARNING_RATE, betas=(config.MOMENTUM, 0.99))
    optim_dis = optim.Adam(net_dis.parameters(), lr=config.LEARNING_RATE, betas=(config.MOMENTUM, 0.99))

    net_gen, net_dis, losses, _, imgs = train_base(config.EPOCHS, config.BATCH_SIZE, config.DIM_NOISE, config.DEVICE,
                                                    dataset, net_gen, net_dis, loss_main, optim_gen, optim_dis, config.PATH_MODEL)
    
    return net_gen, net_dis, losses, imgs

