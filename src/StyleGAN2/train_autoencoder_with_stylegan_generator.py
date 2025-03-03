import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils


import wandb



from src.dataset3d.Dataset3D import UVTextureDataset

from src.GANTrainer import Trainer
from src.StyleGAN2.custom_encoder import CustomEncoder
from src.StyleGAN2.stylegan2_pytorch import Generator, replicate_tensor

from src.set_seed import set_seed

# Set seed for reproducibility
set_seed()


def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).to(device)

class AutoEncoderTrainer(Trainer):

    def __init__(self, uv_textures_directory, lr=0.0001, batch_size=8, latent_vector_size=64, **kwargs):
        """
        Initialize DCWGANTrainerParent with configuration for GAN training.

        Args:
            random_size (int): Size of the random noise vector.
            lr (float): Learning rate for optimizers.
            weight_clipping (tuple): Min and max values for weight clipping (used in WGAN).
            dataroot (str): Directory path for the dataset.
            workers (int): Number of data loading workers.
            batch_size (int): Batch size for training.
            latent_vector_size (int): Size of the latent vector for generator input.
            wgan (bool): Flag to select WGAN or regular GAN.
            **kwargs: Additional arguments for the Trainer class.
        """
        super().__init__(lr=lr,**kwargs)

        self.uv_textures_directory = uv_textures_directory
        self.batch_size = batch_size
        self.latent_vector_size = latent_vector_size
        self.lr = lr
        self.loss_function = nn.MSELoss()
        self.image_size = 128
    
    def init_dataset(self):
        dataset = UVTextureDataset(
            dataset_directory=self.dataset_path,
            uv_textures_directory=self.uv_textures_directory,
            device=self.device,
            image_size=self.image_size
        )

        # Create a DataLoader for the dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True, 
        )

    def init_GAN(self):
        self.E = CustomEncoder(latent_vector_size=self.latent_vector_size).to(self.device)
        self.G = Generator(image_size=self.image_size, latent_dim=self.latent_vector_size).to(self.device)
        self.GAN = self.G
        model_params = list(self.E.parameters()) + list(self.G.parameters())
        self.optimizer = torch.optim.Adam(model_params, lr=self.lr)

    def train_epoch(self):

        self.E.train()
        self.G.train()

        # Iterate over the DataLoader
        for i, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            uv_textures, _ = batch
            encoded_uv_textures = self.E(uv_textures)

            # Replicating for each batch
            encoded_textures_vector = replicate_tensor(encoded_uv_textures, self.G.num_layers)
            noise = image_noise(encoded_textures_vector.size(0), self.image_size, device=self.device)

            # Forward pass through the encoder and decoder
            decoded_textures1 = self.G(encoded_textures_vector, noise)

            # Compute the loss
            loss = self.loss_function(decoded_textures1, uv_textures)

            loss.backward()
            self.optimizer.step()

            decoded_textures_grid1 = vutils.make_grid(decoded_textures1, normalize=True)

            if i % self.evaluate_every == 0:
                wandb.log({"decoded_textures_grid": wandb.Image(decoded_textures_grid1),
                            "uv_textures_grid":  wandb.Image(uv_textures),
                            "MSE loss": loss.item(),
                          }
                         )

if __name__ == "__main__":
    trainer = AutoEncoderTrainer(num_epochs=10,latent_vector_size=512,batch_size=8, uv_textures_directory='./my_data/uv_textures_128', name='model', models_dir='fresh_data/stylegan2_autoencoder')
    trainer.prepare_for_training()
    trainer.train_model()








