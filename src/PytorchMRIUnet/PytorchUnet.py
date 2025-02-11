#!/usr/bin/env python


import torch
import torch.nn as nn
import torch.utils.data

from src.custom_models.discriminators import  DiscriminatorW256
from src.PytorchMRIUnet.Unet import UNet, UnetRandomPart

    


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