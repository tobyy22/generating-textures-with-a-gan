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




from src_c.WGANUnet.WGANUnet import WGANUnet
from src_c.WGANUnet.losses import wasserstein_loss, similarity_loss
from src_c.GANTrainer import Trainer
from src_c.utils import save_tensors_to_png




class WGANUnetTrainer(Trainer):
    def __init__(self, random_size=50, lr=0.0002, similarity_loss=True, similarity_loss_alpha=3, weight_clipping=(-0.01, 0.01), **kwargs):
        super().__init__(lr=lr,**kwargs)
        self.random_size = random_size
        self.similarity_loss = similarity_loss
        self.similarity_loss_alpha = similarity_loss_alpha
        self.weight_clipping = weight_clipping
        


    def train_epoch(self):
        for i, data in enumerate(tqdm(self.dataset)):

            # WGAN discriminator weight clipping
            for parm in self.GAN.discriminator.parameters():
                parm.data.clamp_(
                    self.weight_clipping[0],
                    self.weight_clipping[1],
                )

            # Training the discriminator on real examples
            real_examples = data  # shape [64, 3, 6]
            batch_size = real_examples.size(0)

            # Compute the Wasserstein loss for the real examples
            self.GAN.discriminator_optimizer.zero_grad()
            real_output = self.GAN.discriminator(real_examples).view(-1)
            real_loss = wasserstein_loss(real_output, real_flag=True)
            real_loss.backward()

            # Generate fake textures using the encoder and decoder
            uv_textures = self.dataset.load_current_batch_position_textures()

            generated_textures, uv_textures_encoded, middle_layers = self.generate_fake_texture(uv_textures)

            # Render the current data batch with the generated fake textures
            fake_renders = self.dataset.get_fake_data(generated_textures)

            # Compute the Wasserstein loss for the fake examples
            fake_output = self.GAN.discriminator(fake_renders.detach()).view(-1)
            fake_loss = wasserstein_loss(fake_output, real_flag=False)
            fake_loss.backward()

            # Update the discriminator's weights
            errD = real_loss + fake_loss
            self.GAN.discriminator_optimizer.step()

            # U-net training
            self.GAN.decoder.zero_grad()
            self.GAN.encoder.zero_grad()

            # already have the fake renders produced by the U-net
            output = self.GAN.discriminator(fake_renders).view(-1)

            # why real flag: because we want to know the distance from the 'realness'
            errG_wasserstein = wasserstein_loss(output, real_flag=True)
            errG_wasserstein.backward()


            log_dict = {
                "generator_loss": errG_wasserstein.item(),
                "discriminator_loss": errD.item(),
            }

            if self.similarity_loss:
                other_noises = [
                    torch.randn(
                        batch_size // self.number_of_mesh_views,
                        self.random_size,
                        1,
                        1,
                        device=self.device,
                    )
                    for _ in range(self.similarity_loss_alpha)
                ]
                partly_random_other_noises = [
                    torch.cat((other_noise, uv_textures_encoded), dim=1)
                    for other_noise in other_noises
                ]

                other_fakes = []
                for other_noise in partly_random_other_noises:
                    other_fake = self.GAN.decoder(other_noise, middle_layers)
                    other_fakes.append(other_fake)

                errG_similar_outputs = similarity_loss(other_fakes)
                errG_similar_outputs.backward()
                log_dict["generator_similarity_loss"] = errG_similar_outputs.item()

            self.GAN.generator_optimizer.step()

            # wandb.log(log_dict)

            if i % 50 == 0:
                fid_score = self.compute_fid_score()
                visual_data = self.visualize_data()

                all_logs = {**fid_score, **visual_data, **log_dict}

                wandb.log(all_logs)

        # self.epoch += 1
    
    def generate_fake_texture(self, position_textures):
        uv_textures_encoded, middle_layers = self.GAN.encoder(position_textures)
        noise = torch.randn(
            position_textures.size(0),
            self.random_size,
            1,
            1,
            device=self.device,
        )
        uv_texture_encoded_and_noise_concat = torch.cat(
            (noise, uv_textures_encoded), dim=1
        )
        generated_textures = self.GAN.decoder(
            uv_texture_encoded_and_noise_concat, middle_layers
        )
        return generated_textures, uv_textures_encoded, middle_layers

    def generate_texture_for_object(self, index):
        position_texture = self.dataset.load_specific_position_textures(index=index)
        generated_texture, _, _ = self.generate_fake_texture(position_texture)
        generated_texture = generated_texture[:1, :, :, :]
        return generated_texture



    def visualize_data2(self, index=2):
        with torch.no_grad():

            decoded_texture1 = self.generate_texture_for_object(index)

            rendered_views_using_texture1 = self.dataset.get_multiple_views(
                index, decoded_texture1
            )

            save_tensors_to_png(rendered_views_using_texture1, 'pngs/fake2')

    def init_GAN(self):
        self.GAN = WGANUnet(lr=self.lr, ngpu=self.ngpu)



if __name__ == "__main__":
    agent = WGANUnetTrainer(
                 pre_generated_uv_textures_dir='my_data/uv_textures_64',
                 pregenerate_uv_textures=False,
                 uv_textures_pregenerated=True)
    # agent.load()
    agent.prepare_for_training()
    agent.train_model()

