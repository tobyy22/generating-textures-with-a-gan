import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import wandb
from tqdm import tqdm


from src.DCWGAN.DCWGANTrainerParent import DCWGANTrainerParent
from src.set_seed import set_seed

# Set random seed for reproducibility
set_seed()

class DCWGANTrainer(DCWGANTrainerParent):
    """
    This class is adjusted to train GAN on 'normal' dataset – without any rendering. 
    It will be used for overfitting experiments to see if everything works properly.
    """

    def init_dataset(self):
        """
        This function is rewritten as we can use standard dataset from Tensorflow. 
        """
        self.dataset = dset.ImageFolder(
            root=self.dataroot,
            transform=transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=False,
            num_workers=self.workers,
            batch_size=self.batch_size
        )


    @torch.no_grad()
    def visualize_data(self):
        """
        Visualize generated images from the fixed noise vector.

        Returns:
            dict: Contains a wandb.Image for tracking in wandb.
        """
        self.GAN.eval()
        fake = self.GAN.generator(self.fixed_noise).detach().cpu()
        grid = vutils.make_grid(fake, normalize=True)
        grid = np.transpose(grid.numpy(), (1,2,0))

        return {"view_grid": wandb.Image(grid)}

    def train_epoch(self):
        """
        Training loop for basic GAN. 
        """
        for i, data in tqdm(enumerate(self.dataloader, 0)):
            
            # Apply weight clipping if using WGAN
            if self.wgan:
                for parm in self.GAN.discriminator.parameters():
                    parm.data.clamp_(self.weight_clipping[0], self.weight_clipping[1])
            
            #switch GAN to training mode
            self.GAN.train()

            # Discriminator training on real images
            self.GAN.discriminator.zero_grad()
            real_examples = data[0].to(self.device)
            batch_size = real_examples.size(0)
            output = self.GAN.discriminator(real_examples).view(-1)
            errD_real = self.loss_function(output, real_flag=True, device=self.device)
            errD_real.backward()

            # Discriminator training on fake images
            noise = torch.randn(batch_size, self.latent_vector_size, 1, 1, device=self.device)
            fake = self.GAN.generator(noise)
            output = self.GAN.discriminator(fake.detach()).view(-1)
            errD_fake = self.loss_function(output, real_flag=False, device=self.device)
            errD_fake.backward()
            errD = errD_fake + errD_real
            self.GAN.discriminator_optimizer.step()

            # Generator training
            self.GAN.generator.zero_grad()
            output = self.GAN.discriminator(fake).view(-1)
            errG = self.loss_function(output, real_flag=True, device=self.device)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.GAN.generator_optimizer.step()

            # Logging training metrics
            log_dict = {
                "generator_loss": errG.item(),
                "discriminator_loss": errD.item(),
            }
            
            # Logging visualization data (e.g., generated images) if applicable
            visual_data = self.visualize_data()
            all_logs = {**visual_data, **log_dict}
            wandb.log(all_logs)

if __name__ == "__main__":
    # Instantiate and configure the trainer class
    agent = DCWGANTrainer(
        num_epochs=4000,
        name='model',
        models_dir='my_data/DCWGAN',
<<<<<<< HEAD
        wgan=True,
        image_size=64
=======
        wgan=True
>>>>>>> 2320bde642d217f0717b24db9f1c728d1b28a447
    )
    
    # Prepare model for training and start the training process
    agent.prepare_for_training()
    agent.train_model()
