
import torch.nn as nn

class CustomEncoder(nn.Module):
    def __init__(self, latent_vector_size=64):
        super(CustomEncoder, self).__init__()

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


    def forward(self, x):

        encoded_features = self.encoder(x)            
        flattened = self.flatten(encoded_features)    
        bottleneck = self.encoder_fc(flattened)

        return bottleneck