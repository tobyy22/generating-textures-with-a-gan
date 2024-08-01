#!/usr/bin/env python



import sys
import pathlib

p = pathlib.Path(__file__).parents[1]
sys.path.append(str(p))

import wandb
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.utils.data




from src_c.DCGAN.DCGAN import DCGANExtended
from src_c.WGANUnet.losses import wasserstein_loss, similarity_loss
from src_c.GANTrainer import Trainer
from src_c.utils import save_tensors_to_png




class DCGANExtendedTrainer(Trainer):
    def __init__(self, random_size=50, lr=0.0002, similarity_loss=True, similarity_loss_alpha=3, weight_clipping=(-0.01, 0.01), **kwargs):
        super().__init__(lr=lr,**kwargs)
        self.random_size = random_size
        self.similarity_loss = similarity_loss
        self.similarity_loss_alpha = similarity_loss_alpha
        self.weight_clipping = weight_clipping
        


    def train_epoch(self):
        for i, data in tqdm(enumerate(self.dataset)):

            # WGAN discriminator weight clipping
            for parm in self.GAN.discriminator.parameters():
                parm.data.clamp_(
                    self.weight_clipping[0],
                    self.weight_clipping[1],
                )

            self.netD.zero_grad()

            real_examples = data
            batch_size = real_examples.size(0)
            label = torch.full((batch_size,), self.real_label, dtype=torch.float, device=self.device)
            output = self.GAN.discriminator(real_examples).view(-1)
            errD_real = -torch.mean(output)
            errD_real.backward()
            
            # D_x = output.mean().item()
            # noise = torch.randn(self.number_of_meshes_per_iteration, self.nz, 1, 1, device=self.device)
            # fake = self.netG(noise)



            generated_fake_textures = self.generate_fake_texture()
            label.fill_(self.fake_label)
            
            rendered_fake_views = self.dataset.get_fake_data(generated_fake_textures)
            output = self.netD(rendered_fake_views.detach()).view(-1)

            errD_fake = torch.mean(output)
            errD_fake.backward()
            # D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.GAN.discriminator_optimizer.step()
            


            self.GAN.generator.zero_grad()
            label.fill_(self.real_label)  #

            output = self.GAN.discriminator(rendered_fake_views).view(-1)
            errG = -torch.mean(output)
            errG.backward()
            # D_G_z2 = output.mean().item()
            self.GAN.generator_optimizer.step()

            log_dict = {
                "generator_loss": errG.item(),
                "discriminator_loss": errD.item(),
            }                           
            
            if i % 50 == 0:
                fid_score = self.compute_fid_score()
                visual_data = self.visualize_data()

                all_logs = {**fid_score, **visual_data, **log_dict}

                wandb.log(all_logs)
    
    def generate_fake_texture(self):
        noise = torch.randn(self.number_of_meshes_per_iteration, self.nz, 1, 1, device=self.device)
        generated_textures = self.netG(noise)
        return generated_textures

    def generate_texture_for_object(self, index=None):
        generated_texture, _, _ = self.generate_fake_texture()
        generated_texture = generated_texture[:1, :, :, :]
        return generated_texture



    def init_GAN(self):
        self.GAN = DCGANExtended(lr=self.lr, ngpu=self.ngpu)



if __name__ == "__main__":
    agent = DCGANExtendedTrainer(
                 pre_generated_uv_textures_dir='my_data/uv_textures_64',
                 pregenerate_uv_textures=False,
                 uv_textures_pregenerated=True)
    # agent.load()
    agent.prepare_for_training()
    agent.train_model()

