#!/usr/bin/env python

from __future__ import print_function

import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import wandb

from tqdm import tqdm

import sys

import pathlib
p = pathlib.Path(__file__).parents[1]
sys.path.append(str(p))

from dataset3d.Dataset3D import Dataset3D
from graphs.models.Generator import Generator
from graphs.models.Discriminator import DiscriminatorW
import os



class DCGAN:
    def __init__(self, gan_model = 'my_data/GAN_model'):
        self.dataroot = "/projects/3DDatasets/3D-FUTURE/3D-FUTURE-model"
        self.invalid_data_file = "./my_data/nerenderovatelne.txt"
        self.number_of_meshes_per_iteration = 16
        self.number_of_mesh_views = 4
        self.batch_size = self.number_of_meshes_per_iteration * self.number_of_mesh_views
        self.image_size = 64
        self.nz = 100
        self.num_epochs = 5
        self.learning_rate = 0.0002
        self.ngpu = 1


        manualSeed = random.randint(1, 10000) 
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        self.device = torch.device("cpu")

        self.generator_path = os.path.join(gan_model, 'generator.pt')
        self.discriminator_path = os.path.join(gan_model, 'discriminator.pt')



        if not os.path.exists(self.generator_path) or not os.path.exists(self.discriminator_path):
            print('Initializing new models.')
            self.netG = Generator(self.ngpu).to(self.device)

            if (self.device.type == 'cuda') and (self.ngpu > 1):
                self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))


            self.netG.apply(DCGAN.weights_init)

            self.netD = DiscriminatorW(self.ngpu).to(self.device)

            if (self.device.type == 'cuda') and (self.ngpu > 1):
                self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))

            self.netD.apply(DCGAN.weights_init)

        else:
            print('Loading existing models.')
            self.netG = torch.load(self.generator_path, map_location=self.device)
            if (self.device.type == 'cuda') and (self.ngpu > 1):
                self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))

            self.netD = torch.load(self.discriminator_path, map_location=self.device)
            if (self.device.type == 'cuda') and (self.ngpu > 1):
                self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))

        self.fixed_noise = torch.randn(64, self.nz, 1, 1, device=self.device)

        self.real_label = 1.
        self.fake_label = 0.


        self.dataset = Dataset3D( self.dataroot, 16, 4, self.invalid_data_file, True)

        wandb.init(project="generating-textures-with-a-gan", 
            config = {
                "number_of_meshes_per_iteration" : self.number_of_meshes_per_iteration,
                "number_of_mesh_views": self.number_of_mesh_views,
                "image_size": self.image_size
                }
            )

        self.optimizerD = torch.optim.RMSprop(self.netD.parameters(), lr=self.learning_rate)
        self.optimizerG = torch.optim.RMSprop(self.netG.parameters(), lr=self.learning_rate)

        self.epoch = 0


    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    
    def train_epoch(self):
        for i, data in tqdm(enumerate(self.dataset)):

            for parm in self.netD.parameters():
                    parm.data.clamp_(-0.01, 0.01)

            self.netD.zero_grad()

            real_examples = data
            b_size = real_examples.size(0)

            label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)

            output = self.netD(real_examples).view(-1)
            

            errD_real = -torch.mean(output)
            errD_real.backward()
            D_x = output.mean().item()


            noise = torch.randn(self.number_of_meshes_per_iteration, self.nz, 1, 1, device=self.device)
            fake = self.netG(noise)
            label.fill_(self.fake_label)
            
            fake_examples = self.dataset.get_fake_data(fake)


            output = self.netD(fake_examples.detach()).view(-1)

            errD_fake = torch.mean(output)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.optimizerD.step()
            


            self.netG.zero_grad()
            label.fill_(self.real_label)  #

            output = self.netD(fake_examples).view(-1)
            errG = -torch.mean(output)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.optimizerG.step()

            wandb.log({'generator_loss': errG.item(), 
                        'discriminator_loss': errD.item()
                        }
                    )                           
            
            if i % 50 == 0:
                print(f'Epoch: {self.epoch}/{self.num_epochs}, Batch: {i}/{len(self.dataset)}')
                
                self.visualize_data()

            self.epoch += 1

    def visualize_data(self, index = 2455):

        with torch.no_grad():
            fake = self.netG(self.fixed_noise)
            texture_grid = vutils.make_grid(fake.detach().cpu(), normalize=True)
            test = self.dataset.texture_random_mesh_with_different_textures(fake, i = index).detach().cpu()
            view_grid = vutils.make_grid(test.detach().cpu(), normalize=True)
            wandb.log({'view_grid/': wandb.Image(view_grid), 
                    'texture_grid/': wandb.Image(texture_grid)
                    }
                    )


    def train_model(self):
        print("Starting Training Loop...")
        # For each epoch
        for _ in tqdm(range(self.num_epochs)):
            # For each batch in the dataloader
            print(f'Epoch: {self.epoch}')
            self.train_epoch()
            torch.save(self.netG, self.generator_path)
            torch.save(self.netD, self.discriminator_path)

if __name__ == "__main__":
    agent = DCGAN()
    agent.train_model()
    agent.visualize_data()


    