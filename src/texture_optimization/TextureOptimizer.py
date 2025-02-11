#!/usr/bin/env python

import time
import itertools
import wandb

import torch
import numpy as np

from src.dataset3d.Dataset3D import Dataset3D
from src.set_seed import set_seed

# Set seed for reproducibility
set_seed()



def checkerboard_texture(device):
    texture = np.indices((1024,1024)).sum(axis=0) % 2
    texture = np.repeat(texture[:, :, np.newaxis], 3, axis=2)
    texture = np.expand_dims(texture, axis=0)
    texture = texture.astype(np.float32)
    return torch.tensor(texture, device=device, requires_grad=True)


class TextureOptimizationGridSearch:
    def __init__(
        self,
        dataset_path='./my_data/texture_prediction/dataset_just_cow',
        num_views_per_iteration_to_optimize=20,
        time_to_optimize=300,
        penalization_loss=True,
        param_grid=None,
    ):
        # Assign instance variables from parameters
        self.dataset_path = dataset_path
        self.num_views_per_iteration_to_optimize = num_views_per_iteration_to_optimize
        self.time_to_optimize = time_to_optimize
        self.penalization_loss = penalization_loss

        # Default param grid if not provided
        if param_grid is None:
            param_grid = {
                "learning_rate": [1.0, 0.5, 0.1],
                "loss_rgb": [15000, 30000, 25000],
                "loss_penalization": [0.15, 0.3, 0.5],
                "momentum": [0.9],
            }
        self.param_grid = param_grid


        # Setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Dataset setup
        self.dataset = Dataset3D(
            self.dataset_path,
            1,
            self.num_views_per_iteration_to_optimize,
            device=self.device,
        )

        # Generate permutations from param grid
        keys, values = zip(*self.param_grid.items())
        self.permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Texture setup
        self.texture = checkerboard_texture(self.device)

        # Loss weights setup
        if self.penalization_loss:
            self.loss_weights = {"loss_rgb": None, "loss_penalization": None}
        else:
            self.loss_weights = {"loss_rgb": None}

        # Initialize loss values tensors
        self.rgb_losses_values = torch.empty(
            len(self.permutations_dicts), len(self.dataset), requires_grad=False
        )
        self.penalization_losses_values = torch.empty(
            len(self.permutations_dicts), len(self.dataset), requires_grad=False
        )


    def grid_search(self):

        row_index = 0
        for params in self.permutations_dicts:

            for loss_name in self.loss_weights:
                self.loss_weights[loss_name] = params[loss_name]
   
            learning_rate = params["learning_rate"]
            momentum = params["momentum"]
            column_index = 0            
            
            for data in self.dataset:


                data = torch.permute(data, (0, 2, 3, 1))
                wandb.init(project="generating-textures-with-a-gan", 
                           config = {}
                          )

                origin_texture = checkerboard_texture(self.device) 
                my_optimizer = torch.optim.SGD([origin_texture], lr = learning_rate, momentum=momentum)

                # Compute the losses
                loss_values = {k: torch.tensor(0.0, device=self.device) for k in self.loss_weights}

                # Sum the weighted losses
                losses_values_multiplied = {k: torch.tensor(0.0, device=self.device) for k in self.loss_weights}



                start_time = time.time()
                i = 0
                while time.time() <= start_time + self.time_to_optimize:

                    # Main optimization loop
                    my_optimizer.zero_grad()

                    # Render with the new texture
                    rendered_with_new_texture = self.dataset.get_fake_data_for_texture_optimization(origin_texture)
                    rendered_with_new_texture = torch.permute(rendered_with_new_texture, (0, 2, 3, 1))


                    loss_rgb = torch.mean(((data - rendered_with_new_texture) ** 2), dim=(1, 2, 3))
                    loss_rgb = torch.sum(loss_rgb)
                    loss_values["loss_rgb"] = loss_rgb / self.num_views_per_iteration_to_optimize

                    if self.penalization_loss:
                        # Penalize values below 0
                        penalization_below_0 = origin_texture[origin_texture < 0.0]
                        penalization_below_0 = -torch.sum(penalization_below_0)

                        # Penalize values above 1
                        penalization_above_1 = origin_texture[origin_texture > 1.0]
                        penalization_above_1 = torch.sum(penalization_above_1 - 1)

                        # Add penalties together and normalize
                        penalization = penalization_below_0 + penalization_above_1
                        loss_values["loss_penalization"] = penalization / self.num_views_per_iteration_to_optimize


                    min_texture_value = torch.min(origin_texture)
                    max_texture_value = torch.max(origin_texture)



                    sum_loss = torch.tensor(0.0, device=self.device)
                    for k, l in self.loss_weights.items():
                        loss_value = loss_values[k]
                        loss_weight = self.loss_weights[k]
                        multiplied_loss = loss_value * loss_weight
                        losses_values_multiplied[k] = multiplied_loss
                        sum_loss += multiplied_loss

                    
                    # Convert origin_texture to image
                    origin_texture_image = origin_texture.detach().cpu().numpy()
                    origin_texture_image = (origin_texture_image - origin_texture_image.min()) / (
                        origin_texture_image.max() - origin_texture_image.min()
                    )  # Normalize to [0, 1]
                    origin_texture_image = (origin_texture_image * 255).astype("uint8").squeeze() # Convert to 8-bit

                    log_dict = {
                        "loss_rgb": float(losses_values_multiplied["loss_rgb"].detach().cpu()),
                        "max_texture_value": float(max_texture_value.detach().cpu()),
                        "min_texture_value": float(min_texture_value.detach().cpu()),
                        "origin_texture/": wandb.Image(origin_texture_image)
                            }

                    if self.penalization_loss:                          
                        log_dict["loss_penalization"] = float(losses_values_multiplied["loss_penalization"].detach().cpu())


                    # Log to wandb
                    wandb.log(log_dict)

                    # Optimization step
                    sum_loss.backward()
                    torch.nn.utils.clip_grad_norm_(origin_texture, 0.5)
                    my_optimizer.step()
                    
                    i+=1

                self.rgb_losses_values[row_index, column_index] = losses_values_multiplied["loss_rgb"]
                self.penalization_losses_values[row_index, column_index] = losses_values_multiplied["loss_penalization"]
                
                column_index += 1
                
                
            row_index += 1
            column_index = 0
        

    def compute_best_parameters(self):
        if len(self.permutations_dicts) < 2:
            print('Cannot compute best parameters if there is less than two parameters combinations!')
            return
        
        parameters_list = self.permutations_dicts
        rgb_losses = self.rgb_losses_values
        penalization_losses = self.penalization_losses_values


        normal_rgb = (rgb_losses - torch.min(rgb_losses))/(torch.max(rgb_losses) - torch.min(rgb_losses))
        normal_penal = (penalization_losses - torch.min(penalization_losses))/(torch.max(penalization_losses) - torch.min(penalization_losses))
        normal_sum = normal_rgb + normal_penal

        sum_over_all_data_loss = torch.sum(normal_sum, axis=1)
        best_params_index = torch.argmin(sum_over_all_data_loss, axis=0)

        print(f'Best parameters for texture optimization: {parameters_list[best_params_index.item()]}')




if __name__ == "__main__":
    agent = TextureOptimizationGridSearch()
    agent.grid_search()
    agent.compute_best_parameters()
