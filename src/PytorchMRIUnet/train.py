#!/usr/bin/env python

import copy
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from tqdm import tqdm
import wandb

from src.PytorchMRIUnet.PytorchUnet import PytorchUnet
from src.losses import wasserstein_loss
from src.GANTrainer import Trainer
from src.set_seed import set_seed

# Set seed for reproducibility
set_seed()

def adjust_weights(model, adjust_bottleneck=False, indices=[], noise_std=1e-2):
    """
    Adjusts the weights of specified layers in the model for inference by introducing noise.

    Args:
        model (nn.Module): The model whose weights will be adjusted.
        adjust_bottleneck (bool): If True, applies noise to the bottleneck layer weights.
        indices (list of int): List of layer indices to apply noise (e.g., [1, 2] for encoder1/decoder1 and encoder2/decoder2).
        noise_std (float): Standard deviation of the noise to be added to the weights.
    """
    if adjust_bottleneck:
        for param in model.bottleneck.parameters():
            if param.requires_grad:
                param.data += torch.randn_like(param.data) * noise_std

    layers_to_adjust = {
        1: [model.encoder1, model.decoder1],
        2: [model.encoder2, model.decoder2],
        3: [model.encoder3, model.decoder3],
        4: [model.encoder4, model.decoder4]
    }

    for idx in indices:
        if idx in layers_to_adjust:
            for layer in layers_to_adjust[idx]:
                for param in layer.parameters():
                    if param.requires_grad:
                        param.data += torch.randn_like(param.data) * noise_std

class PytorchUnetTrainer(Trainer):
    """
    Trainer for Pytorch U-Net with WGAN discriminator.

    Attributes:
        lr (float): Learning rate.
        weight_clipping (tuple): Range for discriminator weight clipping in WGAN.
        inference_noise (bool): Whether to add noise during inference.
        noise_std (float): Standard deviation of noise for weights adjustment.
        adjust_bottleneck (bool): Whether to adjust the bottleneck during inference.
        layers_noise (list): Indices of layers to add noise during inference.
    """

    def __init__(self, lr=0.00001, weight_clipping=(-0.01, 0.01), inference_noise=True, 
                 noise_std=1e-2, adjust_bottleneck=False, layers_noise=[1, 2, 3, 4], **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.lr = lr
        self.weight_clipping = weight_clipping
        self.inference_noise = inference_noise
        self.noise_std = noise_std
        self.adjust_bottleneck = adjust_bottleneck
        self.layers_noise = layers_noise

    def train_epoch(self):
        """
        Train the model for one epoch, including the WGAN generator and discriminator.
        """
        for i, data in enumerate(tqdm(self.dataset)):
            # WGAN discriminator weight clipping
            for param in self.GAN.discriminator.parameters():
                param.data.clamp_(*self.weight_clipping)

            self.GAN.train()

            # Training the discriminator with real examples
            real_examples = data
            self.GAN.discriminator_optimizer.zero_grad()
            real_output = self.GAN.discriminator(real_examples).view(-1)
            real_loss = wasserstein_loss(real_output, real_flag=True)
            real_loss.backward()

            uv_textures = self.dataset.load_current_batch_position_textures()

            # Generate fake textures using the encoder and decoder
            generated_textures = self.generate_fake_texture(uv_textures)

            # Render fake data and compute discriminator loss
            fake_renders = self.dataset.get_fake_data(generated_textures)
            fake_output = self.GAN.discriminator(fake_renders.detach()).view(-1)
            fake_loss = wasserstein_loss(fake_output, real_flag=False)
            fake_loss.backward()

            # Update discriminator weights
            errD = real_loss + fake_loss
            self.GAN.discriminator_optimizer.step()

            # U-Net generator training
            self.GAN.generator_optimizer.zero_grad()
            output = self.GAN.discriminator(fake_renders).view(-1)
            errG_wasserstein = wasserstein_loss(output, real_flag=True)
            errG_wasserstein.backward()

            # Logging losses
            log_dict = {
                "generator_loss": errG_wasserstein.item(),
                "discriminator_loss": errD.item(),
            }
            self.GAN.generator_optimizer.step()

            if i % self.evaluate_every == 0:
                visual_data = self.visualize_data(1000)
                visual_data1 = self.visualize_data(1433, postfix='_1')
                all_logs = {**visual_data, **visual_data1, **log_dict}
                wandb.log(all_logs)

    def generate_fake_texture(self, position_textures):
        """
        Generate textures using the generator.

        Args:
            position_textures (Tensor): Input textures to the generator.

        Returns:
            Tensor: Generated textures.
        """
        unet_output, _ = self.GAN.generator(position_textures)
        return unet_output

    def generate_texture_for_object(self, index):
        """
        Generate texture for a specific object with optional inference noise.

        Args:
            index (int): Index of the object to generate texture for.

        Returns:
            Tuple[Tensor, Tensor]: Generated texture and input position texture.
        """
        if self.inference_noise:
            original_weights = copy.deepcopy(self.GAN.generator.state_dict())
            adjust_weights(self.GAN.generator, adjust_bottleneck=self.adjust_bottleneck, 
                           noise_std=self.noise_std, indices=self.layers_noise)

        position_texture = self.dataset.load_specific_position_textures(index=index)
        generated_texture = self.generate_fake_texture(position_texture)
        generated_texture = generated_texture[:1, :, :, :]

        if self.inference_noise:
            self.GAN.generator.load_state_dict(original_weights)

        return generated_texture, position_texture

    def init_GAN(self):
        """
        Initialize the U-Net model as the generator within the WGAN setup.
        """
        self.GAN = PytorchUnet(self.lr, self.device)

if __name__ == "__main__":
    agent = PytorchUnetTrainer(
                 name='model_fresh',
                 models_dir='fresh_data/PytorchMRIUnet',
                 pre_generated_uv_textures_dir='my_data/uv_textures_128',
                 pregenerate_uv_textures=False,
                 uv_textures_pregenerated=True,
                 num_epochs=20,
                 evaluate_every=1,
                 number_of_meshes_per_iteration=32,
                 number_of_mesh_views=2,
                 image_size=256,
                 texture_size=128,
                 inference_noise=False # inference noise should be set to False for training
                 )    
    

    agent.prepare_for_training()
    agent.train_model()
