#!/usr/bin/env python

# Import necessary libraries
import wandb
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torchvision.utils as vutils

# Import custom modules for GAN and losses
from src.DCWGAN.DCWGAN import DCGANExtended, DCWGANExtended
from src.losses import wasserstein_loss, bce_loss
from src.GANTrainer import Trainer


class DCWGANTrainerParent(Trainer):
    """
    This is base class for 2 training scenarios:
    1. Training basic DC(W)GAN without any rendering. 
    2. Training Training DC(W)GAN with renderer.

    Both these scenarios are WITHOUT conditional input. 
    
    This class allows to switch between normal GAN and WGAN easily. 
    """
    def __init__(self, lr=0.0002, weight_clipping=(-0.01, 0.01), dataroot='my_data/overfit_2_examples', workers=1, batch_size=2, latent_vector_size=100, wgan=True, **kwargs):
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
        self.weight_clipping = weight_clipping
        self.batch_size = batch_size
        self.latent_vector_size = latent_vector_size
        self.wgan = wgan
        self.dataroot = dataroot
        self.workers = workers

        # Initialize a fixed noise vector for visualization (e.g., grid display)
        self.fixed_noise = torch.randn(64, self.latent_vector_size, 1, 1, device=self.device) 

        
        if self.wgan:
            self.loss_function = wasserstein_loss
        else:
            self.loss_function = bce_loss 
    

    def init_GAN(self):
        """
        Initialize the GAN model (DCGAN or WGAN) based on the selected mode.
        """
        if self.wgan:
            self.GAN = DCWGANExtended(self.lr, ngpu=self.ngpu)
        else:
            self.GAN = DCGANExtended(lr=self.lr, ngpu=self.ngpu)