
import torch
import torch.nn as nn




class TwoLayerAutoencoder(nn.Module):
    def __init__(self, latent_vector_size=64):
        super(TwoLayerAutoencoder, self).__init__()

        self.input_channels = 3
        self.image_size = 64
        self.latent_vector_size = latent_vector_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)   
        )

        reduced_size = self.image_size // 4
        self.encoded_features = 32 * 4 * reduced_size * reduced_size

        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(self.encoded_features, self.latent_vector_size)
        self.decoder_fc = nn.Linear(self.latent_vector_size, self.encoded_features)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, self.input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):

        encoded_features = self.encoder(x)            
        flattened = self.flatten(encoded_features)    
        bottleneck = self.encoder_fc(flattened)

        decoded_flat = self.decoder_fc(bottleneck) 
        reduced_size = self.image_size // 4
        decoded_features = decoded_flat.view(-1, 32, reduced_size, reduced_size)
        decoded = self.decoder(decoded_features)

        return bottleneck




class TwoEncoderUNet(nn.Module):
    def __init__(self, latent_vector_size=30):
        super(TwoEncoderUNet, self).__init__()

        self.input_channels = 3
        self.image_size = 64
        self.latent_vector_size = latent_vector_size

        # Main Encoder
        self.main_encoder = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Skip Connection Encoder
        self.skip_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=2),
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
                nn.MaxPool2d(kernel_size=2),
            )
        ])

        reduced_size = self.image_size // 4
        self.encoded_features = 32 * reduced_size * reduced_size

        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(self.encoded_features, self.latent_vector_size)
        self.decoder_fc = nn.Linear(self.latent_vector_size, self.encoded_features)

        # Decoder Blocks with Integrated Upsample
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample the feature map
                nn.Conv2d(32 + 32, 16, kernel_size=3, stride=1, padding=1),  # Convolution to match dimensions
                nn.Tanh()
            ),
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample the feature map
                nn.Conv2d(16 + 16, self.input_channels, kernel_size=3, stride=1, padding=1),  # Convolution to match dimensions
                nn.Sigmoid()
            )
        ])

    def forward(self, x_main, x_skip):
        # Main Encoder Path
        main_encoded = self.main_encoder(x_main)

        # Skip Encoder Path
        encoded_features = []
        current_input = x_skip
        for layer in self.skip_encoder:
            current_input = layer(current_input)
            encoded_features.append(current_input)

        flattened = self.flatten(main_encoded)
        bottleneck = self.encoder_fc(flattened)

        # bottleneck = adjust_latent_vector(bottleneck, 0.5)

        decoded_flat = self.decoder_fc(bottleneck)
        reduced_size = self.image_size // 4
        main_decoded_features = decoded_flat.view(-1, 32, reduced_size, reduced_size)

        # Decoder Path with Skip Connections
        current_input = main_decoded_features
        for i in range(len(encoded_features) - 1, -1, -1):
            skip_connection = encoded_features[i]

            # Concatenate and pass through the decoder block
            current_input = torch.cat([current_input, skip_connection], dim=1)
            current_input = self.decoder_blocks[len(encoded_features) - 1 - i](current_input)

        return current_input


def adjust_latent_vector(latent_vectors, fraction):
    """
    Replaces a fraction of each latent vector's elements, starting from the
    second part of the vector, with random values sampled from the mean and std
    of the input tensor.

    Args:
        latent_vectors (torch.Tensor): Input tensor of shape (batch, latent_dim).
        fraction (float): Fraction of elements to replace (e.g., 0.5 for 50%).

    Returns:
        torch.Tensor: Modified latent vectors.
    """
    batch, latent_dim = latent_vectors.shape
    num_replace = int(latent_dim * fraction)  # Number of values to replace

    # Compute mean and standard deviation from the tensor
    mean = latent_vectors.mean()
    std = latent_vectors.std()

    # Create random values to replace
    random_values = torch.normal(mean.item(), std.item(), size=(batch, num_replace), device=latent_vectors.device)

    # Make a copy to modify
    adjusted_latents = latent_vectors.clone()

    # Replace the last `num_replace` values in the second half of each vector
    start_index = latent_dim - num_replace
    adjusted_latents[:, start_index:] = random_values

    return adjusted_latents