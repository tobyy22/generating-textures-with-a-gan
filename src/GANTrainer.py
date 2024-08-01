#!/usr/bin/env python



import torch
import torch.nn.parallel
import torch.utils.data
from pytorch_fid import fid_score


import torchvision.utils as vutils
import wandb

from tqdm import tqdm

import sys

import pathlib
from pathlib import Path

import json



p = pathlib.Path(__file__).parents[1]
sys.path.append(str(p))

from dataset3d.Dataset3D import Dataset3D

from src_c.utils import save_tensors_to_png, ensure_empty_directory



class Trainer:
    def __init__(self, name='model2', 
                 models_dir='my_data/DCGAN', 
                 dataset_path='/projects/3DDatasets/3D-FUTURE/3D-FUTURE-model', 
                 pre_generated_uv_textures_dir=None,
                 pregenerate_uv_textures=False,
                 uv_textures_pregenerated=False,
                 image_size=64, 
                 number_of_meshes_per_iteration=5, 
                 number_of_mesh_views=20, 
                 num_epochs=20, 
                 device='cuda:0',
                 lr=1e-4,
                 ngpu=1,
                 **args) -> None:
        
        self.name = name
        self.models_dir = Path(models_dir)
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.number_of_meshes_per_iteration = number_of_meshes_per_iteration
        self.number_of_mesh_views = number_of_mesh_views
        self.num_epochs=num_epochs
        self.lr = lr
        self.device = device
        self.ngpu=ngpu

        self.pre_generated_uv_textures_dir=pre_generated_uv_textures_dir
        self.pregenerate_uv_textures=pregenerate_uv_textures
        self.uv_textures_pregenerated=uv_textures_pregenerated

        self.config_path = self.models_dir / name / '.config.json'
        
        # self.device = device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")


        self.GAN = None

        wandb.init(project="generating-textures-with-a-gan", config={})
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)


        self.epoch=0
    
    def train_epoch(self):
        pass

    def train_model(self):
        start_epoch = self.epoch
        print(f"Starting Training Loop from epoch {start_epoch}.")
        print(f'Learning rate: {self.lr}')
        for i in tqdm(range(start_epoch,start_epoch+self.num_epochs)):
            self.epoch = i
            self.save()
            print(f"Epoch: {self.epoch}")
            self.train_epoch()
            self.save()

    def generate_fake_texture(self, position_textures=None):
        pass

    def generate_texture_for_object(self, index):
        pass

    def init_GAN(self):
        pass

    def init_dataset(self):
        self.dataset = Dataset3D(
            self.dataset_path,
            self.number_of_meshes_per_iteration,
            self.number_of_mesh_views,
            device='cuda:0',
            invalid_data_file="./my_data/nerenderovatelne.txt",
            GAN_mode=True,
            pre_generated_uv_textures_dir=self.pre_generated_uv_textures_dir,
            pregenerate_uv_textures=self.pregenerate_uv_textures,
            uv_textures_pregenerated=self.uv_textures_pregenerated,
            image_size=self.image_size
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
            # 'version': __version__
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

        self.prepare_for_training()

        load_data = torch.load(model_path)
        self.GAN.load_state_dict(load_data['GAN'])
        self.epoch = name + 1

    
    @torch.no_grad()
    def visualize_data(self, index=2):
        decoded_texture1 = self.generate_texture_for_object(index)
        rendered_views_using_texture1 = self.dataset.get_multiple_views(
            index, decoded_texture1
        )
        view_grid1 = vutils.make_grid(rendered_views_using_texture1, normalize=True)
        texture1 = vutils.make_grid(decoded_texture1, normalize=True)

        return {
                "view_grid1/": wandb.Image(view_grid1),
                "texture1/": wandb.Image(texture1),
            }
    
    @torch.no_grad()
    def compute_fid_score(self):
        self.GAN.eval()
        data_indices = [2, 1001, 4498, 2333]
        all_real_data_tensors = []
        all_fake_data_tensors = []

        for index in data_indices:
            real_data_tensors = self.dataset.render_specific_object(index)
            real_data_tensors = torch.stack(real_data_tensors)
            generated_texture = self.generate_texture_for_object(index)
            fake_data_tensors = self.dataset.get_multiple_views(index, generated_texture)

            # Append tensors to the lists
            all_real_data_tensors.append(real_data_tensors)
            all_fake_data_tensors.append(fake_data_tensors)

        all_real_data_tensors = torch.cat(all_real_data_tensors, dim=0)
        all_fake_data_tensors = torch.cat(all_fake_data_tensors, dim=0)
        
        real_images_path = 'fid_temp/real'
        fake_images_path = 'fid_temp/fake'

        ensure_empty_directory(real_images_path)
        ensure_empty_directory(fake_images_path)

        save_tensors_to_png(all_real_data_tensors, real_images_path)
        save_tensors_to_png(all_fake_data_tensors, fake_images_path)

        fid_value = fid_score.calculate_fid_given_paths([real_images_path, fake_images_path],
                                                batch_size=50,
                                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                                dims=2048)
        
        
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
                'number_of_meshes_per_iteration': self.number_of_meshes_per_iteration, 
                'number_of_mesh_views': self.number_of_mesh_views,
                'lr': self.lr}