import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import wandb



from src.dataset3d.Dataset3D import UVTextureDataset
from src.autoencoder.AutoEncoder import TwoEncoderUNet

from src.GANTrainer import Trainer

from src.set_seed import set_seed

# Set seed for reproducibility
set_seed()


class AutoEncoderTrainer(Trainer):

    def __init__(self, uv_textures_directory, lr=0.0004, batch_size=8, latent_vector_size=30, **kwargs):
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
            shuffle=True,  # Shuffle the dataset for training
        )

        fourth_elem = dataset.load_texture('metal.jpeg')
        self.real_fixed_textures = fourth_elem.unsqueeze(0).repeat(self.batch_size, *([1] * (fourth_elem.dim())))



    def init_GAN(self):
        self.GAN = TwoEncoderUNet(self.latent_vector_size).to(self.device)
        self.model = self.GAN
        self.optimizer = torch.optim.Adam(self.GAN.parameters(), lr=self.lr)

    def train_epoch(self):

        self.model.train()

        # Iterate over the DataLoader
        for i, batch in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            uv_textures, target_textures = batch

            # Forward pass through the encoder and decoder
            decoded_textures1 = self.model(self.real_fixed_textures, uv_textures)


            # Compute the loss
            # loss = self.loss_function(self.real_fixed_textures, target_textures)

            # loss.backward()
            # self.optimizer.step()

            decoded_textures_grid1 = vutils.make_grid(decoded_textures1, normalize=True)
            uv_textures_grid = vutils.make_grid(uv_textures, normalize=True)
            real_textures_grid = vutils.make_grid(target_textures, normalize=True)


            if i % self.evaluate_every == 0:
                wandb.log({"decoded_textures_grid1": wandb.Image(decoded_textures_grid1),
                            "uv_textures_grid":  wandb.Image(uv_textures_grid),
                            "real_textures_grid":  wandb.Image(real_textures_grid),
                            # "MSE loss": loss.item(),
                          }
                         )






if __name__ == "__main__":
    trainer = AutoEncoderTrainer(num_epochs=60,batch_size=8,uv_textures_directory='my_data/uv_textures_64', name='model_backup', models_dir='my_data/TwoEncoderUnet')
    trainer.load()
    trainer.init_dataset()

    trainer.train_model()








