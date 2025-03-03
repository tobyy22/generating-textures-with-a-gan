#!/usr/bin/env python

import torch.nn as nn
from tqdm import tqdm
import wandb

from src.PytorchMRIUnet.PytorchUnet import PytorchUnetRandomPart
from src.losses import wasserstein_loss, similarity_loss_mse
from src.GANTrainer import Trainer
from src.set_seed import set_seed

# Set seed for reproducibility
set_seed()


class PytorchUnetSimilarityLossTrainer(Trainer):
    """
    Trainer for U-Net with similarity loss.

    This trainer modifies the original U-Net training by introducing a similarity loss term, 
    which encourages the generator to produce diverse outputs for slightly perturbed inputs.

    Attributes:
        random_size (int): Size of the random part added to the bottleneck.
        similarity_loss_weight (float): Weight of the similarity loss term.
        lr (float): Learning rate for the optimizer.
        weight_clipping (tuple): Range for discriminator weight clipping in WGAN.
    """
    def __init__(self, random_size=256, similarity_loss_weight=0.001, lr=0.00003, weight_clipping=(-0.01, 0.01), **kwargs):
        super().__init__(lr=lr, **kwargs)
        self.random_size = random_size
        self.similarity_loss_weight = similarity_loss_weight
        self.lr = lr
        self.weight_clipping = weight_clipping

    def train_epoch(self):
        """
        Train the model for one epoch using WGAN and similarity loss.
        """
        for i, data in enumerate(tqdm(self.dataset)):
            # WGAN discriminator weight clipping
            for param in self.GAN.discriminator.parameters():
                param.data.clamp_(*self.weight_clipping)

            self.GAN.train()

            # Training the discriminator with real examples
            real_examples = data
            batch_size = real_examples.size(0)

            self.GAN.discriminator_optimizer.zero_grad()
            real_output = self.GAN.discriminator(real_examples).view(-1)
            real_loss = wasserstein_loss(real_output, real_flag=True)
            real_loss.backward()

            # Generate fake textures using the encoder and decoder
            uv_textures = self.dataset.load_current_batch_position_textures()
            generated_textures1, generated_textures2 = self.generate_fake_texture(uv_textures)

            # Render the current data batch with the generated fake textures
            fake_renders1 = self.dataset.get_fake_data(generated_textures1)
            fake_renders2 = self.dataset.get_fake_data(generated_textures2)

            # Compute the Wasserstein loss for the fake examples
            fake_output = self.GAN.discriminator(fake_renders1.detach()).view(-1)
            fake_loss = wasserstein_loss(fake_output, real_flag=False)
            fake_loss.backward()

            # Update the discriminator's weights
            errD = real_loss + fake_loss
            self.GAN.discriminator_optimizer.step()

            # U-Net generator training
            self.GAN.generator_optimizer.zero_grad()

            # Generator loss calculation
            output = self.GAN.discriminator(fake_renders1).view(-1)
            errG_wasserstein = wasserstein_loss(output, real_flag=True)
            errG_similarity = similarity_loss_mse(fake_renders1, fake_renders2) * self.similarity_loss_weight
            errG = errG_wasserstein + errG_similarity
            errG.backward()

            # Logging losses
            log_dict = {
                "generator_wasserstein_loss": errG_wasserstein.item(),
                "generator_similarity_loss": errG_similarity.item(),
                "generator_loss": errG.item(),
                "discriminator_loss": errD.item(),
            }
            self.GAN.generator_optimizer.step()

            if i % self.evaluate_every == 0:
                visual_data = self.visualize_data(1000)
                all_logs = {**visual_data, **log_dict}
                wandb.log(all_logs)

    def generate_fake_texture(self, position_textures):
        """
        Generate textures using the U-Net generator.

        Args:
            position_textures (Tensor): Input textures for the generator.

        Returns:
            Tuple[Tensor, Tensor]: Two generated outputs (original and perturbed).
        """
        unet_output1, unet_output2 = self.GAN.generator(position_textures)
        return unet_output1, unet_output2

    def generate_texture_for_object(self, index):
        """
        Generate texture for a specific object using the generator.

        Args:
            index (int): Index of the object to generate the texture for.

        Returns:
            Tuple[Tensor, Tensor]: The generated texture and the input position texture.
        """
        position_texture = self.dataset.load_specific_position_textures(index=index)
        generated_texture, _ = self.generate_fake_texture(position_texture)
        generated_texture = generated_texture[:1, :, :, :]
        return generated_texture, position_texture

    def init_GAN(self):
        """
        Initialize the U-Net model with random parts for similarity loss training.
        """
        self.GAN = PytorchUnetRandomPart(self.lr, self.device, self.random_size)


if __name__ == "__main__":
    agent = PytorchUnetSimilarityLossTrainer(
                 name='model_fresh',
                 models_dir='fresh_data/PytorchMRIUnet',
                 pre_generated_uv_textures_dir='my_data/uv_textures_128',
                 pregenerate_uv_textures=False,
                 uv_textures_pregenerated=True,
                 num_epochs=5,
                 evaluate_every=1,
                 number_of_meshes_per_iteration=32,
                 number_of_mesh_views=2,
                 image_size=256,
                 texture_size=128
                 )    
    
    agent.prepare_for_training()
    agent.train_model()


    # #1433, 1444

    # data = agent.visualize_data(888)
    # wandb.log(data)

    # Save initial weights
    # original_weights = copy.deepcopy(agent.GAN.generator.state_dict())

    # textures =[]

    # for i in range(20):

    #     texture, _ = agent.generate_texture_for_object(888)
    #     textures.append(texture)
    #     # data = agent.visualize_data(888)
    #     # wandb.log(data)

    #     # Perform visualization


    # # Concatenate tensors into a single batch
    # batch = torch.cat(textures, dim=0)  # Shape: [8, 3, 128, 128]

    # # Create a grid
    # grid = make_grid(batch, nrow=5, normalize=True, scale_each=True)  # Arrange 4 images per row

    # # Convert the grid to a PIL Image
    # grid_image = grid.permute(1, 2, 0).mul(255).cpu().numpy()
    # import numpy as np  # Convert to [H, W, C] and scale to [0, 255]
    # grid_image = Image.fromarray(grid_image.astype(np.uint8))

    # # Save the grid to a PNG file
    # grid_image.save("grid_output.png")

