#!/usr/bin/env python


import torch
import torch.nn as nn
import torch.utils.data


from src.PytorchMRIUnet_plg.Unet import UNet, UnetRandomPart


class DiscriminatorW256(nn.Module):
    def __init__(self, ngpu):
        super(DiscriminatorW256, self).__init__()

        ndf = 64
        nc = 3

        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            # Output shape: (1) x 1 x 1
        )

    def forward(self, x):
        return self.main(x)





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
            # nn.Sigmoid() – removed sigmoid layer
        )

    def forward(self, x):
        return self.main(x)
    


class PytorchUnet(nn.Module):
    def __init__(self, lr, device, ngpu=1):
        super().__init__()

        self.lr = lr
        self.ngpu = ngpu
        self.device = device


        """Generator"""
        self.generator = UNet().to(self.device)

        
        """Discriminator"""
        self.discriminator = DiscriminatorW256(self.ngpu).to(self.device)
        self.discriminator.apply(PytorchUnet.weights_init)

        """Optimizers"""
        self.generator_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr*0.8)


    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class PytorchUnetRandomPart(nn.Module):
    def __init__(self, lr, device, random_part=256, ngpu=1):
        super().__init__()

        self.lr = lr
        self.ngpu = ngpu
        self.device = device


        """Generator"""
        self.generator = UnetRandomPart(num_random_channels=random_part).to(self.device)

        
        """Discriminator"""
        self.discriminator = DiscriminatorW256(self.ngpu).to(self.device)
        self.discriminator.apply(PytorchUnet.weights_init)

        """Optimizers"""
        self.generator_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)


    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)