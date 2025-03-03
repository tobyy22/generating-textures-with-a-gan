import torch
import torchvision.utils as vutils
import wandb
from tqdm import tqdm

from src.set_seed import set_seed
from src.DCWGAN.DCWGANTrainerParent import DCWGANTrainerParent

from src.DCWGAN.DCWGAN import DCWGANExtendedHigherResolution




# Set seed for reproducibility
set_seed()

class DCWGANRendererTrainer(DCWGANTrainerParent):
    """
    Trainer class for a DC(W)GAN model with a renderer.
    This trainer is designed for scenarios where both real and fake data involve rendered 3D textures and views.
    """
    
    def train_epoch(self):
        """
        Trains the GAN model for one epoch.

        - Assumes the dataset is an instance of a custom Dataset3D class that provides real, rendered views of 3D objects.
        - Implements a training loop similar to the one in DCWGANTrainer, but adapted for 3D rendering.
        """
        
        # real_examples are a batch of rendered views from the real dataset (provided by Dataset3D).
        for i, real_examples in tqdm(enumerate(self.dataset)):
            
            # WGAN weight clipping for discriminator if in WGAN mode
            if self.wgan:
                for parm in self.GAN.discriminator.parameters():
                    parm.data.clamp_(self.weight_clipping[0], self.weight_clipping[1])

            self.GAN.train()

            # Zero gradients for discriminator and train on real examples
            self.GAN.discriminator.zero_grad()
            output = self.GAN.discriminator(real_examples).view(-1)
            errD_real = self.loss_function(output, real_flag=True)
            errD_real.backward()
            
            # Generate a batch of fake textures
            generated_fake_textures, _ = self.generate_fake_texture()
            
            # Render views of objects from generated fake textures
            rendered_fake_views = self.dataset.get_fake_data(generated_fake_textures)
            output = self.GAN.discriminator(rendered_fake_views.detach()).view(-1)
            errD_fake = self.loss_function(output, real_flag=False)
            errD_fake.backward()
            errD = errD_real + errD_fake
            self.GAN.discriminator_optimizer.step()
            
            # Zero gradients for generator and update
            self.GAN.generator.zero_grad()
            output = self.GAN.discriminator(rendered_fake_views).view(-1)
            errG = self.loss_function(output, real_flag=True)
            errG.backward() 
            self.GAN.generator_optimizer.step()

            # Log generator and discriminator losses
            log_dict = {
                "generator_loss": errG.item(),
                "discriminator_loss": errD.item(),
            }                           
            
            # Periodically log visual data, and other metrics
            if i % self.evaluate_every == 0:
                visual_data = self.visualize_data()
                all_logs = {**visual_data, **log_dict}
                wandb.log(all_logs)
    
    
    def generate_fake_texture(self, uv_textures=None):
        """
        Generate fake textures for each object in the current batch.

        Returns:
            torch.Tensor: Batch of generated textures.
        """
        noise = torch.randn(self.number_of_meshes_per_iteration, self.latent_vector_size, 1, 1, device=self.device)
        generated_textures = self.GAN.generator(noise)
        return generated_textures, uv_textures
    
    
    def generate_texture_for_object(self, index=None):
        """
        Generate a single texture for a specific object by index.

        Args:
            index (int, optional): Index of the object to generate texture for. Defaults to None. 
                                   The argument is kept as we are overriding the method with this,
                                   however it is not used in the function as the texture generation
                                   is independent of the object in this scenario. 

        Returns:
            torch.Tensor: Generated texture for the specified object.
        """
        generated_texture, _ = self.generate_fake_texture()
        # Select only one texture 
        generated_texture = generated_texture[:1, :, :, :]  
        return generated_texture, None

    def init_GAN(self):
        self.GAN = DCWGANExtendedHigherResolution(self.lr, self.device)

    

if __name__ == "__main__":
    agent = DCWGANRendererTrainer(
        name='model',
        models_dir='fresh_data/WGANRendererHighResolution',
        wgan=True,
        num_epochs=5,
        evaluate_every=50,
        number_of_meshes_per_iteration=32,
        number_of_mesh_views=2,
        image_size=256,
        texture_size=128,
        lr=0.0002
    )
    agent.prepare_for_training()
    agent.train_model()





