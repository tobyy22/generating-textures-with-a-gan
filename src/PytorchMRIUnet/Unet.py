from collections import OrderedDict

import torch
import torch.nn as nn



"""
Original U-Net from Pytorch implementation.
"""

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        self.num_random_channels = 0
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16 + self.num_random_channels, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = add_random_channels(bottleneck, self.num_random_channels)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1)), bottleneck

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


def add_random_channels(bottleneck, num_channels):
    """
    Adds a specified number of random channels to the bottleneck tensor.
    
    Args:
        bottleneck (torch.Tensor): The input bottleneck tensor of shape [N, C, H, W].
        num_channels (int): The number of random channels to add.
    
    Returns:
        torch.Tensor: The updated tensor with additional random channels.
    """
    # Ensure the bottleneck tensor is a 4D tensor
    if bottleneck.dim() != 4:
        raise ValueError("Bottleneck tensor must have 4 dimensions (N, C, H, W).")
    
    # Extract the shape of the bottleneck tensor
    batch_size, current_channels, height, width = bottleneck.shape
    
    # Generate random data with the same spatial dimensions
    random_data = torch.randn(batch_size, num_channels, height, width, device=bottleneck.device)
    
    # Concatenate the random data along the channel dimension
    updated_bottleneck = torch.cat((bottleneck, random_data), dim=1)
    
    return updated_bottleneck


"""
A modified version of the original U-Net architecture with added randomness.

This class implements an adjusted U-Net model where a random part is introduced in the bottleneck layer 
to enhance the model's robustness and improve generalization by introducing controlled randomness. 
The bottleneck layer is perturbed with additional random channels, which are used to decode both 
the original bottleneck and the perturbed version separately, producing two outputs: the original 
decoded image and the perturbed decoded image that are then compared using similarity loss function.
"""

class UnetRandomPart(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, init_features=32, num_random_channels=256):
        self.num_random_channels = num_random_channels
        super(UnetRandomPart, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16 + self.num_random_channels, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        # Add random channels to bottleneck
        bottleneck_with_random = add_random_channels(bottleneck, self.num_random_channels)
        perturbed_bottleneck = add_random_channels(bottleneck, self.num_random_channels)

        # Decode the original bottleneck
        dec4 = self.upconv4(bottleneck_with_random)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        output_original = torch.sigmoid(self.conv(self.decoder1(dec1)))

        # Decode the perturbed bottleneck
        dec4_perturbed = self.upconv4(perturbed_bottleneck)
        dec4_perturbed = torch.cat((dec4_perturbed, enc4), dim=1)
        dec4_perturbed = self.decoder4(dec4_perturbed)
        dec3_perturbed = self.upconv3(dec4_perturbed)
        dec3_perturbed = torch.cat((dec3_perturbed, enc3), dim=1)
        dec3_perturbed = self.decoder3(dec3_perturbed)
        dec2_perturbed = self.upconv2(dec3_perturbed)
        dec2_perturbed = torch.cat((dec2_perturbed, enc2), dim=1)
        dec2_perturbed = self.decoder2(dec2_perturbed)
        dec1_perturbed = self.upconv1(dec2_perturbed)
        dec1_perturbed = torch.cat((dec1_perturbed, enc1), dim=1)
        output_perturbed = torch.sigmoid(self.conv(self.decoder1(dec1_perturbed)))

        return output_original, output_perturbed

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )