#!/usr/bin/env python


import torch
import torch.nn as nn
import torch.utils.data

from src.custom_models.discriminators import DiscriminatorW



class Encoder(nn.Module):
    def __init__(self, ngpu, non_random_part_size=50):
        super(Encoder, self).__init__()

        ndf = 64
        nc = 3

        self.ngpu = ngpu
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.relu1=nn.LeakyReLU(0.2, inplace=False)
            # state size. (ndf) x 32 x 32
        self.conv2=nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
        self.batch1=nn.BatchNorm2d(ndf * 2)
        self.relu2=nn.LeakyReLU(0.2, inplace=False)
            # state size. (ndf*2) x 16 x 16
        self.conv3=nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
        self.batch2=nn.BatchNorm2d(ndf * 4)
        self.relu3=nn.LeakyReLU(0.2, inplace=False)
            # state size. (ndf*4) x 8 x 8
        self.conv4=nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
        self.batch3=nn.BatchNorm2d(ndf * 8)
        self.relu4=nn.LeakyReLU(0.2, inplace=False)
            # state size. (ndf*8) x 4 x 4
        self.conv5=nn.Conv2d(ndf * 8, non_random_part_size, 4, 1, 0, bias=False)
        self.tan = nn.Tanh()

    def forward(self, x):
        outs = []
        x=self.conv1(x)
        x=self.relu1(x)
        outs.append(x)
            # state size. (ndf) x 32 x 32
        x=self.conv2(x)
        x=self.batch1(x)
        x=self.relu2(x)
        outs.append(x)
            # state size. (ndf*2) x 16 x 16
        x=self.conv3(x)
        x=self.batch2(x)
        x=self.relu3(x)
        outs.append(x)
            # state size. (ndf*4) x 8 x 8
        x=self.conv4(x)
        x=self.batch3(x)
        x=self.relu4(x)
        outs.append(x)
            # state size. (ndf*8) x 4 x 4
        x=self.conv5(x)
        x=self.tan(x)

        return x,outs


class Decoder(nn.Module):
    def __init__(self, ngpu, latent_vector_size=100):
        super(Decoder, self).__init__()

        nz = latent_vector_size
        ngf = 64
        nc = 3


        self.ngpu = ngpu

        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.batch1=nn.BatchNorm2d(ngf * 8)
        self.relu1=nn.ReLU(False)
            # state size. (ndf) x 32 x 32
        self.conv2=nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1, bias=False)
        self.batch2=nn.BatchNorm2d(ngf * 4)
        self.relu2=nn.ReLU(False)
        

        self.conv3=nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1, bias=False)
        self.batch3=nn.BatchNorm2d(ngf * 2)
        self.relu3=nn.ReLU(False)
            # state size. (ndf*4) x 8 x 8
        self.conv4=nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1, bias=False)
        self.batch4=nn.BatchNorm2d(ngf)
        self.relu4=nn.ReLU(False)
            # state size. (ndf*8) x 4 x 4
        self.conv5=nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False)
        self.tan = nn.Tanh()

    def forward(self, x, skip_connections):

        x=self.conv1(x)
        x=self.batch1(x)
        x=self.relu1(x)

        x = torch.cat((x, skip_connections[-1]), dim=1)
    
        x=self.conv2(x)
        x=self.batch2(x)
        x=self.relu2(x)


        x = torch.cat((x, skip_connections[-2]), dim=1)

        x=self.conv3(x)
        x=self.batch3(x)
        x=self.relu3(x)

        x = torch.cat((x, skip_connections[-3]), dim=1)

        x=self.conv4(x)
        x=self.batch4(x)
        x=self.relu4(x)

        x = torch.cat((x, skip_connections[-4]), dim=1)

        x=self.conv5(x)
        x = self.tan(x)

        return x


class WGANUnet(nn.Module):
    def __init__(self, lr, ngpu, device, latent_vector_size=100, non_random_part_size=50):
        super().__init__()

        self.lr = lr
        self.ngpu = ngpu
        self.device = device


        """Generator"""
        self.encoder = Encoder(self.ngpu, non_random_part_size=non_random_part_size).to(self.device)
        self.encoder.apply(WGANUnet.weights_init)
        self.decoder = Decoder(self.ngpu, latent_vector_size=latent_vector_size).to(self.device)
        self.decoder.apply(WGANUnet.weights_init)
        """Discriminator"""
        self.discriminator = DiscriminatorW(self.ngpu).to(self.device)
        self.discriminator.apply(WGANUnet.weights_init)

        """Optimizers"""
        generator_parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.generator_optimizer = torch.optim.RMSprop(generator_parameters, lr=self.lr)
        self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)


    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)