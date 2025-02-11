import torch.nn as nn

class Discriminator(nn.Module):
    """
    DC(W)GAN Discriminator Network

    This network classifies images as real or fake by downsampling
    input images through a series of convolutional layers, ultimately
    outputting a single value indicating the "realness" of the image.
    """
    def __init__(self, ngpu):
        """
        Initializes the Discriminator network.

        Args:
            ngpu (int): Number of GPUs available. If >1, data parallelism is applied.
        """
        super(Discriminator, self).__init__()

        # Feature map size and color channels
        ndf = 64  # Size of feature maps in discriminator
        nc = 3    # Number of color channels in the input image (RGB)

        self.ngpu = ngpu
        
        # Main network: a series of Conv2d layers for downsampling
        self.main = nn.Sequential(
            # Input: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (ndf) x 32 x 32

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (ndf*2) x 16 x 16

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (ndf*4) x 8 x 8

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: (ndf*8) x 4 x 4

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: (1) x 1 x 1 (single probability score)
        )

    def forward(self, input):
        """
        Forward pass through the discriminator network.

        Args:
            input (torch.Tensor): Input image of shape (batch_size, nc, 64, 64).

        Returns:
            torch.Tensor: Probability score indicating realness of the input image.
        """
        return self.main(input).view(-1, 1).squeeze(1)
    

class DiscriminatorW(nn.Module):
    """
    Wasserstein GAN Discriminator

    Changes from the previous Discriminator:
    - **Sigmoid Activation Removed**: The final sigmoid layer is omitted because WGAN does not interpret discriminator outputs as probabilities.

    """
    def __init__(self, ngpu):
        super(DiscriminatorW, self).__init__()

        ndf = 64
        nc = 3

        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)
    

class DiscriminatorW256(nn.Module):
    def __init__(self, ngpu):
        super(DiscriminatorW256, self).__init__()

        ndf = 64
        nc = 3

        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=False),
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            # Output shape: (1) x 1 x 1
        )

    def forward(self, x):
        return self.main(x)