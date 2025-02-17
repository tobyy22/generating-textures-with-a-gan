#!/usr/bin/env python





import wandb
from tqdm import tqdm

import torch
import torch.nn.parallel
import torch.utils.data




from src.WGANUnet.WGANUnet import WGANUnet
from src.losses import wasserstein_loss
from src.GANTrainer import Trainer


from src.set_seed import set_seed

# Set seed for reproducibility
set_seed()




class WGANUnetTrainer(Trainer):
    def __init__(self, latent_vector_size=100, random_size=50, lr=0.0002, weight_clipping=(-0.01, 0.01), **kwargs):
        super().__init__(lr=lr,**kwargs)
        self.latent_vector_size = latent_vector_size
        self.random_size = random_size
        self.non_random_size = latent_vector_size - random_size
        self.weight_clipping = weight_clipping

        
    def train_epoch(self):
        for i, data in enumerate(tqdm(self.dataset)):

            # WGAN discriminator weight clipping
            for parm in self.GAN.discriminator.parameters():
                parm.data.clamp_(
                    self.weight_clipping[0],
                    self.weight_clipping[1],
                )

            self.GAN.train()
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


            self.GAN.generator_optimizer.step()

            if i % self.evaluate_every == 0:
                visual_data = self.visualize_data()
                all_logs = {**visual_data, **log_dict}
                wandb.log(all_logs)
    
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
        return generated_texture, position_texture
    

    def init_GAN(self):
        self.GAN = WGANUnet(lr=self.lr, ngpu=self.ngpu, device=self.device, latent_vector_size=self.latent_vector_size, non_random_part_size=self.non_random_size)



if __name__ == "__main__":
    agent = WGANUnetTrainer(
                 name='model',
                 models_dir='my_data/WGANUnet',
                 pre_generated_uv_textures_dir='my_data/uv_textures_64',
                 uv_textures_pregenerated=True,
                 image_size=64,
                 texture_size=64,
                 number_of_meshes_per_iteration=5, 
                 number_of_mesh_views=20, 
                 num_epochs=20,
                 evaluate_every=50, 
                 lr=0.0002,
                 )    
    
    agent.prepare_for_training()
    agent.train_model()


