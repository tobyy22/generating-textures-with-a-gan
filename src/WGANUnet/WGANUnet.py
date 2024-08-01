#!/usr/bin/env python

import sys
import pathlib

p = pathlib.Path(__file__).parents[1]
sys.path.append(str(p))

import torch
import torch.nn as nn
import torch.utils.data


"""TOOD: Rewrite it similary to stylegan2 implementation
put gans files into one file
generate_fake_texture function
function naming
"""

class DiscriminatorW(nn.Module):
    def __init__(self, ngpu):
        super(DiscriminatorW, self).__init__()

        ndf = 64
        nc = 3

        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    

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
            # nn.Sigmoid()

    def forward(self, input):
        outs = []
        input=self.conv1(input)
        input=self.relu1(input)
        outs.append(input)
            # state size. (ndf) x 32 x 32
        input=self.conv2(input)
        input=self.batch1(input)
        input=self.relu2(input)
        outs.append(input)
            # state size. (ndf*2) x 16 x 16
        input=self.conv3(input)
        input=self.batch2(input)
        input=self.relu3(input)
        outs.append(input)
            # state size. (ndf*4) x 8 x 8
        input=self.conv4(input)
        input=self.batch3(input)
        input=self.relu4(input)
        outs.append(input)
            # state size. (ndf*8) x 4 x 4
        input=self.conv5(input)

        return input,outs


class Decoder(nn.Module):
    def __init__(self, ngpu, latent_vector_size=100):
        super(Decoder, self).__init__()

        nz = latent_vector_size
        ngf = 64
        nc = 3


        self.ngpu = ngpu

        self.conv1 = nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False)
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

    def forward(self, input, middle_layers):
        input=self.conv1(input)
        input=self.batch1(input)
        input=self.relu1(input)

        input = torch.cat((input, middle_layers[-1]), dim=1)
    
        input=self.conv2(input)
        input=self.batch2(input)
        input=self.relu2(input)


        input = torch.cat((input, middle_layers[-2]), dim=1)


        input=self.conv3(input)
        input=self.batch3(input)
        input=self.relu3(input)

        input = torch.cat((input, middle_layers[-3]), dim=1)

        input=self.conv4(input)
        input=self.batch4(input)
        input=self.relu4(input)

        input = torch.cat((input, middle_layers[-4]), dim=1)

        input=self.conv5(input)
        input = self.tan(input)

        return input





class WGANUnet(nn.Module):
    def __init__(self, lr=0.0002, ngpu=1, device='cuda:0'):
        super().__init__()

        self.lr = lr
        self.ngpu = ngpu
        self.device = device


        """Generator"""
        self.encoder = Encoder(self.ngpu).to(self.device)
        self.encoder.apply(WGANUnet.weights_init)
        self.decoder = Decoder(self.ngpu).to(self.device)
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