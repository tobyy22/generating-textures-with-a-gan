#!/usr/bin/env python

import time
import itertools
import wandb

import torch
import torchvision.utils as vutils
import numpy as np

from src.dataset3d.Dataset3D import Dataset3D
from src.set_seed import set_seed

# Set seed for reproducibility
set_seed()

def checkerboard_texture(device):
    """
    Generates a 128x128 checkerboard texture as a PyTorch tensor.
    
    Args:
        device (torch.device): The device to allocate the tensor to.
    
    Returns:
        torch.Tensor: A 3-channel checkerboard texture tensor.
    """
    texture = np.indices((128,128)).sum(axis=0) % 2
    texture = np.repeat(texture[:, :, np.newaxis], 3, axis=2)
    texture = np.expand_dims(texture, axis=0)
    texture = texture.astype(np.float32)
    return torch.tensor(texture, device=device, requires_grad=True)


class TextureOptimizationGridSearch:
    """
    A class to perform grid search optimization for texture generation.
    """
    
    def __init__(
        self,
        dataset_path='./my_data/texture_prediction/dataset_just_cow',
        num_views_per_iteration_to_optimize=20,
        time_to_optimize=300,
        penalization_loss=True,
        param_grid=None,
    ):
        """
        Initializes the TextureOptimizationGridSearch class with given parameters.
        
        Args:
            dataset_path (str): Path to the dataset.
            num_views_per_iteration_to_optimize (int): Number of views per iteration.
            time_to_optimize (int): Time allocated for optimization.
            penalization_loss (bool): Whether to apply penalization loss.
            param_grid (dict, optional): Hyperparameter grid for optimization.
        """
        self.dataset_path = dataset_path
        self.num_views_per_iteration_to_optimize = num_views_per_iteration_to_optimize
        self.time_to_optimize = time_to_optimize
        self.penalization_loss = penalization_loss

        if param_grid is None:
            param_grid = {
                "learning_rate": [1.0, 0.5, 0.1],
                "loss_rgb": [15000, 30000, 25000],
                "loss_penalization": [0.15, 0.3, 0.5],
                "momentum": [0.9],
            }
        self.param_grid = param_grid

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dataset = Dataset3D(
            self.dataset_path,
            1,
            self.num_views_per_iteration_to_optimize,
            device=self.device,
            image_size=256,
            texture_size=128
        )

        keys, values = zip(*self.param_grid.items())
        self.permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        self.texture = checkerboard_texture(self.device)
        self.loss_weights = {"loss_rgb": None, "loss_penalization": None} if self.penalization_loss else {"loss_rgb": None}
        self.rgb_losses_values = torch.empty(len(self.permutations_dicts), len(self.dataset), requires_grad=False)
        self.penalization_losses_values = torch.empty(len(self.permutations_dicts), len(self.dataset), requires_grad=False)

    def grid_search(self):
        """
        Performs grid search over the parameter space to optimize texture generation.
        """
        row_index = 0
        for params in self.permutations_dicts:
            for loss_name in self.loss_weights:
                self.loss_weights[loss_name] = params[loss_name]
   
            learning_rate = params["learning_rate"]
            momentum = params["momentum"]
            column_index = 0            
            
            for data in self.dataset:
                # data are views rendered with real texture
                data = torch.permute(data, (0, 2, 3, 1))
                wandb.init(project="generating-textures-with-a-gan", config={})

                origin_texture = checkerboard_texture(self.device) 
                my_optimizer = torch.optim.SGD([origin_texture], lr=learning_rate, momentum=momentum)
                
                start_time = time.time()
                while time.time() <= start_time + self.time_to_optimize:
                    my_optimizer.zero_grad()
                    rendered_with_new_texture = self.dataset.get_fake_data_for_texture_optimization(origin_texture)
                    rendered_with_new_texture = torch.permute(rendered_with_new_texture, (0, 2, 3, 1))
                    # Compute RGB loss between renders (fake x real texture)
                    loss_rgb = torch.mean(((data - rendered_with_new_texture) ** 2), dim=(1, 2, 3))
                    loss_rgb = torch.sum(loss_rgb) / self.num_views_per_iteration_to_optimize
                    loss_values = {"loss_rgb": loss_rgb}
                    
                    if self.penalization_loss:
                        # Compute distances of all pixels from RGB range(0, 1) 
                        penalization = -torch.sum(origin_texture[origin_texture < 0.0]) + torch.sum(origin_texture[origin_texture > 1.0] - 1)
                        loss_values["loss_penalization"] = penalization / self.num_views_per_iteration_to_optimize
                    
                    sum_loss = sum(loss_values[k] * self.loss_weights[k] for k in self.loss_weights)
                    sum_loss.backward()
                    torch.nn.utils.clip_grad_norm_(origin_texture, 0.5)
                    my_optimizer.step()
                
                column_index += 1
            row_index += 1

    def compute_best_parameters(self):
        """
        Computes the best parameter combination based on loss minimization.
        """
        if len(self.permutations_dicts) < 2:
            print('Cannot compute best parameters if there is less than two parameter combinations!')
            return
        
        normal_rgb = (self.rgb_losses_values - torch.min(self.rgb_losses_values)) / (torch.max(self.rgb_losses_values) - torch.min(self.rgb_losses_values))
        normal_penal = (self.penalization_losses_values - torch.min(self.penalization_losses_values)) / (torch.max(self.penalization_losses_values) - torch.min(self.penalization_losses_values))
        normal_sum = normal_rgb + normal_penal
        best_params_index = torch.argmin(torch.sum(normal_sum, axis=1), axis=0)
        print(f'Best parameters for texture optimization: {self.permutations_dicts[best_params_index.item()]}')
