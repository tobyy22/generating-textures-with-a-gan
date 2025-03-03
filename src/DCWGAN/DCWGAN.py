import torch
import torch.nn as nn

from src.custom_models.discriminators import Discriminator, DiscriminatorW, DiscriminatorW256



def weights_init(m):
    """
    Weights initialization function from the PyTorch DCGAN project, 
    forming the basis of this architecture.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


"""
This module includes definitions of the neural networks for DC(W)GAN.
"""

class Generator(nn.Module):
    """
    DC(W)GAN Generator Network

    This network generates synthetic images from a latent vector (noise).
    The architecture uses a series of transposed convolutional layers 
    to upsample the input noise into an image.
    """
    def __init__(self, ngpu):
        """
        Initializes the Generator network.

        Args:
            ngpu (int): Number of GPUs available. If >1, data parallelism is applied.
        """
        super(Generator, self).__init__()
        
        # Latent vector (input noise) size, feature map sizes, and color channels
        nz = 100  # Size of the latent vector
        ngf = 64  # Size of feature maps in generator
        nc = 3    # Number of color channels in the output image (RGB)

        self.ngpu = ngpu
        
        # Main network: a series of ConvTranspose2d layers for upsampling
        self.main = nn.Sequential(
            # Input is Z (latent vector), shape: (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Shape: (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Shape: (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Shape: (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Shape: (ngf) x 32 x 32
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output shape: (nc) x 64 x 64
        )

    def forward(self, input):
        """
        Forward pass through the generator network.

        Args:
            input (torch.Tensor): Latent vector (noise) of shape (batch_size, nz, 1, 1).

        Returns:
            torch.Tensor: Generated image of shape (batch_size, nc, 64, 64).
        """
        return self.main(input)


class Generator128(nn.Module):
    """
    DC(W)GAN Generator Network

    This network generates synthetic images from a latent vector (noise).
    The architecture uses a series of transposed convolutional layers 
    to upsample the input noise into an image.
    """
    def __init__(self, ngpu):
        """
        Initializes the Generator network.

        Args:
            ngpu (int): Number of GPUs available. If >1, data parallelism is applied.
        """
        super(Generator128, self).__init__()

        # Latent vector (input noise) size, feature map sizes, and color channels
        nz = 100  # Size of the latent vector
        ngf = 64  # Size of feature maps in generator
        nc = 3    # Number of color channels in the output image (RGB)

        self.ngpu = ngpu

        # Main network: a series of ConvTranspose2d layers for upsampling
        self.main = nn.Sequential(
            # Input is Z (latent vector), shape: (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Shape: (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Shape: (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Shape: (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Shape: (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # Shape: (ngf//2) x 64 x 64

            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output shape: (nc) x 128 x 128
        )

    def forward(self, input):
        return self.main(input)




class DCGANExtended(nn.Module):
    """
    Wrapper class for DCGAN models with integrated optimizers.

    Key Features:
    - **Generator & Discriminator Initialization**: Instantiates and initializes the Generator and Discriminator, applying the `weights_init` method.
    - **Optimizers**: Configures separate Adam optimizers for the generator and discriminator, with customizable learning rate (`lr`) and beta parameters (`betas`).
    """
    
    def __init__(self, lr=0.0002, betas=(0.5,0.999), ngpu=1, device='cuda:0'):
        super().__init__()

        self.lr = lr
        self.betas = betas
        self.ngpu = ngpu
        self.device = device

        # Generator setup
        self.generator = Generator(self.ngpu).to(self.device)
        self.generator.apply(weights_init)
        
        # Discriminator setup
        self.discriminator = Discriminator(self.ngpu).to(self.device)
        self.discriminator.apply(weights_init)

        # Optimizers
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr/13, betas=self.betas)



class DCWGANExtended(nn.Module):
    """
    Similar class as DCGANExtended with the WGAN adjustments. 
    """
    def __init__(self, lr=0.0002, ngpu=1, device='cuda:0'):
        super().__init__()

        self.lr = 0.0002
        self.ngpu = ngpu
        self.device = device


        # Generator setup
        self.generator = Generator(self.ngpu).to(self.device)
        self.generator.apply(weights_init)
        
        #Discriminator setup
        self.discriminator = DiscriminatorW(self.ngpu).to(self.device)
        self.discriminator.apply(weights_init)

        # Optimizers
        self.generator_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=0.00008)


class DCWGANExtendedHigherResolution(nn.Module):
    """
    Similar class as DCGANExtended with the WGAN adjustments and higher resolution networks. 
    """
    def __init__(self, lr=0.0002, ngpu=1, device='cuda:0'):
        super().__init__()

        self.lr = lr
        self.ngpu = ngpu
        self.device = device


        # Generator setup
        self.generator = Generator128(self.ngpu).to(self.device)
        self.generator.apply(weights_init)
        
        #Discriminator setup
        self.discriminator = DiscriminatorW256(self.ngpu).to(self.device)
        self.discriminator.apply(weights_init)

        # Optimizers
        self.generator_optimizer = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

