#!/usr/bin/env python


import json
import numpy as np
from pathlib import Path

import torch
import torch.nn.parallel
import torch.utils.data

import torchvision.utils as vutils
from tqdm import tqdm
import wandb

# FID computation
from pytorch_fid import fid_score

# Custom modules
from src.dataset3d.Dataset3D import Dataset3D
from src.utils.helper_functions import save_tensors_to_png, ensure_empty_directory




class Trainer:
    def __init__(self, 
                 name, 
                 models_dir, 
                 dataset_path='./3DDataset', 
                 pre_generated_uv_textures_dir=None,
                 pregenerate_uv_textures=False,
                 uv_textures_pregenerated=False,
                 image_size=256, 
                 texture_size=128,
                 number_of_meshes_per_iteration=16, 
                 number_of_mesh_views=4, 
                 num_epochs=20,
                 evaluate_every=50, 
                 lr=1e-4,
                 ngpu=1,
                 **args):
        
        """
        Initializes the Trainer class for running an experiment, setting up paths, model parameters, 
        and configuration for dataset, textures, and training.

        Args:
            name (str): The name of the model or experiment.
            models_dir (str): Directory where all models for the experiment will be saved.
            dataset_path (str, optional): Path to the dataset (default is './3DDataset').
            pre_generated_uv_textures_dir (str, optional): Path to pre-generated UV textures (default is None).
            pregenerate_uv_textures (bool, optional): Flag to indicate whether UV textures should be pregenerated (default is False).
            uv_textures_pregenerated (bool, optional): Flag indicating if UV textures have already been pregenerated (default is False).
            image_size (int, optional): Size of the rendered views to be used in the experiment (default is 256).
            texture_size (int, optional): Size of the UV textures (default is 128).
            number_of_meshes_per_iteration (int, optional): Number of meshes processed per iteration (default is 16).
            number_of_mesh_views (int, optional): Number of views generated per mesh (default is 4).
            num_epochs (int, optional): Number of epochs for training (default is 20).
            evaluate_every (int, optional): Frequency (in terms of epochs) to perform evaluation in wandb (default is 50).
            lr (float, optional): Learning rate for the optimizer (default is 1e-4).
            ngpu (int, optional): Number of GPUs to use for training (default is 1).
            **args: Additional arguments passed to the parent class or specific experiment configurations.

        """
        
        self.name = name
        self.models_dir = Path(models_dir)
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.texture_size = texture_size
        self.number_of_meshes_per_iteration = number_of_meshes_per_iteration
        self.number_of_mesh_views = number_of_mesh_views
        self.num_epochs=num_epochs
        self.evaluate_every = evaluate_every
        self.lr = lr
        self.ngpu=ngpu

        self.pre_generated_uv_textures_dir=pre_generated_uv_textures_dir
        self.pregenerate_uv_textures=pregenerate_uv_textures
        self.uv_textures_pregenerated=uv_textures_pregenerated

        self.config_path = self.models_dir / name / '.config.json'
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        
        self.GAN = None

        wandb.init(project="generating-textures-with-a-gan", config={})
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)


        self.epoch=0
    
    """
    The following three methods should be implemented for each experiment.
    """
    def train_epoch(self):
        raise NotImplementedError

    def generate_texture_for_object(self, index):
         raise NotImplementedError

    def init_GAN(self):
         raise NotImplementedError


    """
    Basic training loop.
    """
    def train_model(self):
        start_epoch = self.epoch
        print(f"Starting Training Loop from epoch {start_epoch}.")
        self.GAN.train()
        for i in tqdm(range(start_epoch,start_epoch+self.num_epochs+1)):
            self.epoch = i
            print(f"Epoch: {self.epoch}")
            self.train_epoch()
            self.save()



    def init_dataset(self):
        self.dataset = Dataset3D(
            self.dataset_path,
            self.number_of_meshes_per_iteration,
            self.number_of_mesh_views,
            device=self.device,
            invalid_data_file="./my_data/nerenderovatelne.txt",
            pre_generated_uv_textures_dir=self.pre_generated_uv_textures_dir,
            pregenerate_uv_textures=self.pregenerate_uv_textures,
            uv_textures_pregenerated=self.uv_textures_pregenerated,
            image_size=self.image_size,
            texture_size=self.texture_size
        )

    def prepare_for_training(self):
        self.init_dataset()
        self.init_GAN()

    def model_path(self, epoch):
        return str(self.models_dir / self.name / f'model_{epoch}.pt')
    

    def save(self):
        assert self.GAN is not None
        save_data = {
            'GAN': self.GAN.state_dict(),
        }

        torch.save(save_data, self.model_path(self.epoch))
        self.write_config()

    def load(self, num = -1):
        self.load_config()
        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
        print(f'continuing from previous epoch - {name}')

        model_path = self.models_dir / self.name / f"model_{name}.pt"

        self.init_GAN()

        load_data = torch.load(model_path, weights_only=True)
        self.GAN.load_state_dict(load_data['GAN'])
        self.GAN.to(self.device)
        self.epoch = name + 1

    
    @torch.no_grad()
    def visualize_data(self, index=888, normalize=False):
        """
        Visualizes real and fake data along with rendered views to Weights & Biases (wandb).

        This method sets the model to evaluation mode, generates a texture for the given object index,
        renders multiple views using both the generated and real textures, and logs the visualizations 
        to Weights & Biases (wandb).

        Args:
            index (int, optional): The index of the object in the dataset to visualize (default: 888).
            normalize (bool, optional): Whether to normalize the images when creating grids (default: False).

        Returns:
            dict: A dictionary containing images logged to wandb with the following keys:
                - `"fake_grid/"`: Rendered views using the generated texture.
                - `"fake_texture/"`: The generated texture itself.
                - `"real_grid/"`: Rendered views using the real texture.
                - `"position_texture/"` (if available): The positional texture visualization.
        """
        self.GAN.eval()
        decoded_texture1, position_texture = self.generate_texture_for_object(index)
        rendered_views_using_texture1 = self.dataset.get_multiple_views(
            index, decoded_texture1
        )
        view_grid1 = vutils.make_grid(rendered_views_using_texture1, normalize=normalize)
        texture1 = vutils.make_grid(decoded_texture1, normalize=normalize)
        
        # Handle case when position_texture is None
        if position_texture is not None:
            position_texture = vutils.make_grid(position_texture, normalize=normalize)
        else:
            position_texture = None  # Placeholder or skip it

        real_views = self.dataset.get_multiple_views(index)
        real_views_grid = vutils.make_grid(real_views, normalize=normalize)

        result = {
            f"fake_grid/": wandb.Image(view_grid1),
            f"fake_texture/": wandb.Image(texture1),
            f"real_grid/": wandb.Image(real_views_grid),
        }

        if position_texture is not None:
            result[f"position_texture/"] = wandb.Image(position_texture)

        return result
    
    
    @torch.no_grad()
    def compute_fid_score(self, generation_batch_size=10):
    
        """
        Computes the Fréchet Inception Distance (FID) score to evaluate the similarity between
        generated images from a GAN and real images. The function allows generating images in 
        batches to prevent memory overflow by saving images as PNGs temporarily. The FID score 
        is calculated on these saved images.

        Args:
            generation_batch_size : (int, default=10): The batch size for generating and saving images, 
                                                    used to manage memory consumption.

        Returns:
            dict: A dictionary containing the calculated FID score.

        """
        self.GAN.eval()
        
        real_images_path = 'fid_temp3/real'
        fake_images_path = 'fid_temp3/fake'
        
        ensure_empty_directory(real_images_path)
        ensure_empty_directory(fake_images_path)

        data_indices = list(range(len(self.dataset.dataloader.data_names)))

        for start_idx in range(0, len(data_indices), generation_batch_size):
            batch_indices = data_indices[start_idx:start_idx + generation_batch_size]
            batch_real_data_tensors = []
            batch_fake_data_tensors = []

            for index in batch_indices:
                real_data_tensors = self.dataset.render_specific_object(index)
                real_data_tensors = torch.stack(real_data_tensors)
                generated_texture, _ = self.generate_texture_for_object(index)
                fake_data_tensors = self.dataset.get_multiple_views(index, generated_texture)

                # Append tensors to the lists
                batch_real_data_tensors.append(real_data_tensors)
                batch_fake_data_tensors.append(fake_data_tensors)

            # Concatenate the batch tensors and save them as PNGs
            batch_real_data_tensors = torch.cat(batch_real_data_tensors, dim=0)
            batch_fake_data_tensors = torch.cat(batch_fake_data_tensors, dim=0)
            
            # Save this batch to PNGs
            save_tensors_to_png(batch_real_data_tensors, real_images_path, start_index=start_idx*20)
            save_tensors_to_png(batch_fake_data_tensors, fake_images_path, start_index=start_idx*20)


        # Now calculate FID score from all saved PNG images
        fid_value = fid_score.calculate_fid_given_paths(
            [real_images_path, fake_images_path],
            batch_size=50,
            device=self.device,
            dims=2048
        )
        
        return {'fid_score': fid_value.item()}
        
    

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.number_of_meshes_per_iteration = config['number_of_meshes_per_iteration']
        self.number_of_mesh_views = config['number_of_mesh_views']
        self.lr = config['lr']


    def config(self):
        return {'image_size': self.image_size, 
                'texture_size': self.texture_size,
                'number_of_meshes_per_iteration': self.number_of_meshes_per_iteration, 
                'number_of_mesh_views': self.number_of_mesh_views,
                'lr': self.lr}