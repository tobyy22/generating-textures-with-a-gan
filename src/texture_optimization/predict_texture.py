#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


# # Fit a mesh via rendering
# 
# This tutorial shows how to:
# - Load a mesh and textures from an `.obj` file. 
# - Create a synthetic dataset by rendering a textured mesh from multiple viewpoints
# - Fit a mesh to the observed synthetic images using differential silhouette rendering
# - Fit a mesh and its textures using differential textured rendering

# ## 0. Install and Import modules

# Ensure `torch` and `torchvision` are installed. If `pytorch3d` is not installed, install it using the following cell:

# In[1]:


import json
from collections import defaultdict
import time
import itertools
import os
import sys
from unittest import result
import torch
import shutil
need_pytorch3d=False
try:
    import pytorch3d
except ModuleNotFoundError:
    need_pytorch3d=True
if need_pytorch3d:
    if torch.__version__.startswith("1.10.") and sys.platform.startswith("linux"):
        # We try to install PyTorch3D via a released wheel.
        pyt_version_str=torch.__version__.split("+")[0].replace(".", "")
        version_str="".join([
            f"py3{sys.version_info.minor}_cu",
            torch.version.cuda.replace(".",""),
            f"_pyt{pyt_version_str}"
        ])
        get_ipython().system('pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/{version_str}/download.html')
    else:
        # We try to install PyTorch3D from source.
        get_ipython().system('curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz')
        get_ipython().system('tar xzf 1.10.0.tar.gz')
        os.environ["CUB_HOME"] = os.getcwd() + "/cub-1.10.0"
        get_ipython().system("pip install 'git+https://github.com/facebookresearch/pytorch3d.git'")


# In[2]:


import os
import torch
import matplotlib.pyplot as plt


import pathlib
p = pathlib.Path(__file__).parents[1]
sys.path.append(str(p))

from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm.notebook import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    AmbientLights
)

# add path for demo utils functions 
import sys
import os
import wandb


def visualize_texture(tex):
    to_vis = torch.squeeze(tex)
    # plt.imsave('../../k1/k.png', to_vis.detach().cpu().numpy())
    plt.imshow(to_vis.detach().cpu().numpy())
    plt.show()

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
device = torch.device("cpu")



def checkerboard_texture(device):
    texture = np.indices((1024,1024)).sum(axis=0) % 2
    texture = np.repeat(texture[:, :, np.newaxis], 3, axis=2)
    texture = np.expand_dims(texture, axis=0)
    texture = texture.astype(np.float32)
    return torch.tensor(texture, device=device, requires_grad=True)

from dataset3d.Dataset3D import Dataset3D
from render.RenderWrap import RenderWrap

class TextureOptimizationGridSearch:
    def __init__(self):
        self.path_to_full_dataset = "/projects/3DDatasets/3D-FUTURE/3D-FUTURE-model"
        self.filtered_dataset = './my_data/texture_prediction/dataset'
        self.num_views_per_iteration_to_optimize = 20
        self.time_to_optimize = 180

        

        self.filter_categories()

        self.dataset = Dataset3D(self.filtered_dataset, 
                                1, 
                                self.num_views_per_iteration_to_optimize,
                                GAN_mode=False
                                )
        

        # self.param_grid = {"learning_rate" : [1.0, 0.5], 
        #      "momentum" : [0.9, 0.0], 
        #      "loss_rgb" : [15000, 30000, 25000], 
        #      "loss_penalization" : [0.15, 0.30, 0.5]
        #      }

        self.param_grid = {"learning_rate" : [1.0], 
             "momentum" : [0.9], 
             "loss_rgb" : [15000], 
             "loss_penalization" : [0.15]
             }
        
        self.texture = checkerboard_texture(device)

        self.losses = {"rgb": {"weight": 1., "value": 0.}, "penalization": {"weight": 1., "value": 0. }}

        self.create_and_save_grid_combinations()


        self.losses = {"rgb": {"weight": 1., "values": []}, "penalization": {"weight": 1., "values": []}}



    def create_and_save_grid_combinations(self):
        keys, values = zip(*self.param_grid.items())
        self.permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]


        self.results = torch.empty(len(self.permutations_dicts), len(self.losses) * len(self.dataset), requires_grad=False)

        with open("./my_data/texture_prediction/grid_search/param_dict.json", "w") as grid_combinations_file:
            json.dump(self.permutations_dicts, grid_combinations_file)

    

    def filter_categories(self, super_category=True):
        category = 'super-category' if super_category else 'category'
        data = None

        model_info = os.path.join(self.path_to_full_dataset, 'model_info.json')

        with open(model_info) as f:
            data = json.load(f)

        categories = set()

        for i in data:
            if i[category] not in categories: 
                categories.add(i[category])
                shutil.copytree(os.path.join(self.path_to_full_dataset, i["model_id"]), 
                                os.path.join(self.filtered_dataset, i["model_id"]), dirs_exist_ok=True)




    def grid_search(self):


        row_index = 0
        for params in self.permutations_dicts:
                
            self.losses["rgb"]["weight"] = params["loss_rgb"]
            self.losses["penalization"]["weight"]  = params["loss_penalization"]
            learning_rate = params["learning_rate"]
            momentum = params["momentum"]
            column_index = 0

            print(params)
            
            
            for data in self.dataset:


                data = torch.permute(data, (0, 2, 3, 1))

                # wandb.init(project="generating-textures-with-a-gan", 
                #            config = {
                #               "object" : "test2",
                #               "learning_rate": learning_rate,
                #               "momentum:": momentum,
                #               "epochs": Niter,
                #               "loss_rgb_mult": loss_rgb_mult, 
                #               "loss_penalization_mult": loss_penalization_mult,
                #               "data_name" : data
                #             }
                #           )

                origin_texture = checkerboard_texture(device) 
                my_optimizer = torch.optim.SGD([origin_texture], lr = learning_rate, momentum=momentum)



                start_time = time.time()
                i = 0
                while time.time() <= start_time + self.time_to_optimize:

                    my_optimizer.zero_grad()


                    rendered_with_new_texture = self.dataset.get_fake_data2(origin_texture)


                    rendered_with_new_texture = torch.permute(rendered_with_new_texture, (0, 2, 3, 1))

                    loss = {k: torch.tensor(0.0, device=device) for k in self.losses}




                    loss_rgb = torch.mean(((data - rendered_with_new_texture) ** 2), dim=(1,2,3))
                    loss_rgb = torch.sum(loss_rgb)
                    loss["rgb"] = loss_rgb / self.num_views_per_iteration_to_optimize

                    penalization_bellow_0 = origin_texture[origin_texture<0.0]
                    penalization_bellow_0 = -torch.sum(penalization_bellow_0)

                    #compute loss of values above 1
                    penalization_above_1 = origin_texture[origin_texture>1.0]
                    penalization_above_1 = torch.sum(penalization_above_1 - 1)

                    #add those two losses together and normalize
                    penalization = penalization_bellow_0 + penalization_above_1
                    loss["penalization"]= penalization/self.num_views_per_iteration_to_optimize

                    

                    sum_loss = torch.tensor(0.0, device=device)
                    for k, l in loss.items():
                        sum_loss += l * self.losses[k]["weight"]
                        self.losses[k]["values"].append(float(l.detach().cpu()))


                    # Print the losses
                    print("total_loss = %.6f" % sum_loss)


                    # Optimization step
                    sum_loss.backward()
                    torch.nn.utils.clip_grad_norm_(origin_texture, 0.5)
                    my_optimizer.step()

                    if torch.isinf(sum_loss).item():
                        print('Infinity loss')
                        break

                    i += 1

                self.results[row_index, column_index] = loss["rgb"]
                self.results[row_index, column_index + 1] = loss["penalization"]
                
                column_index += 2
                
                
            row_index += 1
            column_index = 0
        
        torch.save(self.results, './my_data/texture_prediction/grid_search/results.pt')

    def compute_best_parameters(self):

        results = torch.load('my_data/texture_prediction/grid_search/results.pt')
        parameters_list = json.loads(open('my_data/texture_prediction/grid_search/param_dict.json', 'r').read())

        print(results.size())


        indices_rgb = list(range(0,2,2))
        indices_penal = list(range(1,2,2))


        rgb_losses = results[:,indices_rgb]
        penalization_losses = results[:, indices_penal]


        normal_rgb = (rgb_losses - torch.min(rgb_losses))/(torch.max(rgb_losses) - torch.min(rgb_losses))
        normal_penal = (penalization_losses - torch.min(penalization_losses))/(torch.max(penalization_losses) - torch.min(penalization_losses))

        normal_sum = normal_rgb + normal_penal

        sum_over_all_data_loss = torch.sum(normal_sum, axis=1)

        best_params_index = torch.argmin(sum_over_all_data_loss, axis=0)

        print(f'Metric 1: {parameters_list[best_params_index.item()]}')

        lowest_losses_args = torch.argmin(results, axis=1)

        most_freq = torch.mode(lowest_losses_args)[0].item()
        print(f'Metric 2: {parameters_list[most_freq]}')




if __name__ == "__main__":
    agent = TextureOptimizationGridSearch()
    agent.grid_search()
    agent.compute_best_parameters()




def homogeneous_texture(value, device):
    return torch.full((1, 1024,1024, 3), value, device=device, requires_grad=True)

def uniform_distribution_texture(device):
    texture = torch.rand((1,1024,1024,3), device=device, requires_grad=True)
    return texture
    
    
def visualize_texture(tex):
    to_vis = torch.squeeze(tex)
    plt.imshow(to_vis.detach().cpu().numpy())
    plt.show()
    


# def visualize_target(i):
#     ith_rgb_img = target_rgb[i]
#     plt.imshow(ith_rgb_img.cpu().detach().numpy())
#     plt.show()

# def visualize_mesh(cam_id=1):
#     test = renderer_textured(mesh, cameras=target_cameras[cam_id], lights=lights)
#     plt.imshow(torch.squeeze(test).cpu().detach().numpy()[..., :3])
#     plt.show()


# origin_texture = checkerboard_texture(device)
# visualize_texture(origin_texture)


# # Optimization - hyperparameter tuning.


# # Number of views to optimize over in each SGD iteration
# num_views_per_iteration = num_views
# # Number of optimization steps

# Niter = 100000

# exec_time = 180

# learning_rate = 1.0
# momentum = 0.9

# loss_rgb_mult = 30000.0
# loss_penalization_mult = 0.27

# '''
# LOSS function

# rgb loss: computes MSE between target view and view rendered with predicted texture

# penalization loss: the optimization might be ambiguous, probably because the final rasterization is averageing colours, so the colors 
# in texture can have RGB components out of range 0-1 for floats, so I use also loss to penalize values out of desired rgb range

# '''

# losses = {"rgb": {"weight": loss_rgb_mult, "values": []}, "penalization": {"weight": loss_penalization_mult, "values": []}}

# def texture_and_optimizer():
#     texture = checkerboard_texture(device)
#     optimizer = torch.optim.SGD([texture], lr = learning_rate, momentum=momentum)
#     return texture, optimizer

# '''
# Parameter grid

# All combinations of parameters will be tried. Time complexity grows exponentially with number of parameters. 

# '''
# param_grid = {
#              "loss_rgb" : [30000, 25000, 40000, 50000, 100000], 
#              "loss_penalization" : [0.3, 0.5, 0.4, 0.8, 1]
#              "learning_rate": [1.0],
#              "momentum": [0.9]
#              }


# keys, values = zip(*param_grid.items())
# permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]


# results = torch.empty(len(permutations_dicts), len(losses) * len(data_to_load), requires_grad=False)

# a_file = open("param_dict.json", "w")
# json.dump(permutations_dicts, a_file)
# a_file.close()





# # In[ ]:



# row_index = 0


# for params in permutations_dicts:
#     loss_rgb_mult = params["loss_rgb"]
#     loss_penalization_mult  = params["loss_penalization"]
#     # learning_rate = params["learning_rate"]
#     # momentum = params["momentum"]
#     column_index = 0
    
#     print(params)
    
#     for data in data_to_load:
        
#         print(data)
#         outF = open("checkpoints.txt", "a")
#         outF.write(data)
#         outF.write('\n')
#         outF.close()

#         loop = tqdm(range(Niter))

#         # wandb.init(project="generating-textures-with-a-gan", 
#         #            config = {
#         #               "object" : "test2",
#         #               "learning_rate": learning_rate,
#         #               "momentum:": momentum,
#         #               "epochs": Niter,
#         #               "loss_rgb_mult": loss_rgb_mult, 
#         #               "loss_penalization_mult": loss_penalization_mult,
#         #               "data_name" : data
#         #             }
#         #           )

#         origin_texture, my_optimizer = texture_and_optimizer()

#         mesh, target_images, target_cameras = load_mesh_and_target_rgb(data)


#         target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
#         start_time = time.time()
        

#         for i in loop:
#             my_optimizer.zero_grad()
            
#             #mesh namapuji na novou texturu
#             mesh.textures._maps_padded = origin_texture


#             # Losses to smooth /regularize the mesh shape
#             #tohle jen vytvori tensor ke kazdemu druhu loss
#             #{'rgb': tensor(0., device='cuda:0'), 'silhouette': tensor(0., device='cuda:0'), 'edge': tensor(0., device='cuda:0'), 
#             #'normal': tensor(0., device='cuda:0'), 'laplacian': tensor(0., device='cuda:0')}
#             loss = {k: torch.tensor(0.0, device=device) for k in losses}


#             # update_mesh_shape_prior_losses(mesh, loss)

#             # Randomly select two views to optimize over in this iteration.  Compared
#             # to using just one view, this helps resolve ambiguities between updating
#             # mesh shape vs. updating mesh texture
#             for j in np.random.permutation(num_views).tolist():

#                 #vyrendderuji j-ty pohled
#                 images_predicted = renderer_textured(mesh, cameras=target_cameras[j], lights=lights)

#                 # Squared L2 distance between the predicted RGB image and the target 
#                 # image from our dataset

#                 #odeberu alfa dimenzi
#                 predicted_rgb = images_predicted[..., :3]


#                 #compute rgb rgb loss
#                 loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
#                 loss["rgb"] += loss_rgb / num_views_per_iteration

#                 #compute loss of values below 0
#                 penalization_bellow_0 = origin_texture[origin_texture<0.0]
#                 penalization_bellow_0 = -torch.sum(penalization_bellow_0)

#                 #compute loss of values above 1
#                 penalization_above_1 = origin_texture[origin_texture>1.0]
#                 penalization_above_1 = torch.sum(penalization_above_1 - 1)

#                 #add those two losses together and normalize
#                 penalization = penalization_bellow_0 + penalization_above_1
#                 loss["penalization"] = penalization/num_views_per_iteration


#             #Weighted sum of the losses
#             sum_loss = torch.tensor(0.0, device=device)
#             for k, l in loss.items():
#                 sum_loss += l * losses[k]["weight"]
#                 losses[k]["values"].append(float(l.detach().cpu()))

#             # wandb.log({"loss_rgb": loss["rgb"], "loss_penalization": loss["penalization"], "sum_loss": sum_loss.item()})


#             # Print the losses
#             loop.set_description("total_loss = %.6f" % sum_loss)


#             # Optimization step
#             sum_loss.backward()
#             torch.nn.utils.clip_grad_norm_(origin_texture, 0.5)
#             my_optimizer.step()

#             if time.time() - start_time > exec_time:
#                 print('Time over')
#                 break

#             if torch.isinf(sum_loss).item():
#                 print('Infinity loss')
#                 break
                
#         results[row_index, column_index] = loss["rgb"]
#         results[row_index, column_index + 1] = loss["penalization"]
        
#         column_index += 2
        
#         print(column_index)
        
#     row_index += 1
#     column_index = 0
    
# torch.save(results, 'results.pt')




# # In[ ]:


# results = torch.load('my_data/grid_search/results.pt')
# parameters_list = json.loads(open('my_data/grid_search/param_dict.json', 'r').read())


# # In[ ]:



# indices_rgb = list(range(0,18,2))
# indices_penal = list(range(1,18,2))


# rgb_losses = results[:,indices_rgb]
# penalization_losses = results[:, indices_penal]

# print(rgb_losses)



# normal_rgb = (rgb_losses - torch.min(rgb_losses))/(torch.max(rgb_losses) - torch.min(rgb_losses))
# normal_penal = (penalization_losses - torch.min(penalization_losses))/(torch.max(penalization_losses) - torch.min(penalization_losses))

# normal_sum = normal_rgb + normal_penal

# sum_over_all_data_loss = torch.sum(normal_sum, axis=1)

# best_params_index = torch.argmin(sum_over_all_data_loss, axis=0)

# print(parameters_list[best_params_index.item()])


# # In[ ]:


# lowest_losses_args = torch.argmin(results, axis=1)

# most_freq = torch.mode(lowest_losses_args)[0].item()
# print(parameters_list[most_freq])





# # In[ ]:





# # Optimization settings for ground truth

# # In[ ]:



# mesh, target_images, target_cameras = load_mesh_and_target_rgb('my_data/dataset/cow_mesh/normalized_model.obj')
# target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
# origin_texture, my_optimizer = texture_and_optimizer()




# # Number of views to optimize over in each SGD iteration
# num_views_per_iteration = len(target_rgb)
# # Number of optimization steps

# Niter = 300000

# # Plot period for the losses, saving textures
# plot_period = 30

# #max number of iterations when loss has not decreased at least by the variable below
# #when there is no progress in loss for 'iterations_no_best_loss', I assume the loss has converged
# iterations_no_best_loss = 500

# #loss has to decrease by at least byt this rate, otherwise covergence is assumed
# loss_increased_at_least_by = 0.99

# learning_rate = 1.0
# momentum = 0.9

# loss_rgb_mult = 30000.0
# loss_penalization_mult = 0.3

# '''
# LOSS function:

# rgb loss: computes MSE between target view and view rendered with predicted texture
# penalization: the optimization might be ambiguous, probably because the final rasterization is averageing colours, so the colors 
# in texture can have RGB components out of range 0-1 for floats, so I use also loss to penalize values out of desired rgb range

# '''
# losses = {"rgb": {"weight": loss_rgb_mult, "values": []}, "penalization": {"weight": loss_penalization_mult, "values": []}}


# #optimizing only the texture
# #zkusim vic nahodnych pohledu nebo vypnout momoentum





# # In[ ]:



# loop = tqdm(range(Niter))
# current_iterations_without_best_loss = 0
# best_loss = torch.tensor(float('inf'))
# # wandb.init(project="generating-textures-with-a-gan", config = {
# #   "learning_rate": learning_rate,
# #   "momentum:": momentum,
# #   "epochs": Niter,
# #   "loss_rgb_mult": loss_rgb_mult, 
# #   "loss_penalization_mult": loss_penalization_mult,
# # })



# # store_data = "my_data/predicted/00e4b7dd-08a6-4433-a439-856e4b5de58a/"

# # try:
# #     os.mkdir(store_data)
# # except:
# #   print("Directory exists.")

# # try: 
# #     os.mkdir(store_data + 'textures')
# # except:
# #   print("Directory exists.")


# for i in loop:
#     my_optimizer.zero_grad()
    
#     #mesh namapuji na novou texturu
#     mesh.textures._maps_padded = origin_texture

    
#     # Losses to smooth /regularize the mesh shape
#     #tohle jen vytvori tensor ke kazdemu druhu loss
#     #{'rgb': tensor(0., device='cuda:0'), 'silhouette': tensor(0., device='cuda:0'), 'edge': tensor(0., device='cuda:0'), 
#     #'normal': tensor(0., device='cuda:0'), 'laplacian': tensor(0., device='cuda:0')}
#     loss = {k: torch.tensor(0.0, device=device) for k in losses}
    

#     # update_mesh_shape_prior_losses(mesh, loss)

#     # Randomly select two views to optimize over in this iteration.  Compared
#     # to using just one view, this helps resolve ambiguities between updating
#     # mesh shape vs. updating mesh texture
#     for j in np.random.permutation(num_views).tolist():
        
#         #vyrendderuji j-ty pohled
#         images_predicted = renderer_textured(mesh, cameras=target_cameras[j], lights=lights)

#         # Squared L2 distance between the predicted RGB image and the target 
#         # image from our dataset
        
#         #odeberu alfa dimenzi
#         predicted_rgb = images_predicted[..., :3]
                                         
                                        
        
#         #spocitam rgb loss
#         loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
#         loss["rgb"] += loss_rgb / num_views_per_iteration
        
        
#         penalization_bellow_0 = origin_texture[origin_texture<0.0]
#         penalization_bellow_0 = -torch.sum(penalization_bellow_0)
        
#         penalization_above_1 = origin_texture[origin_texture>1.0]
#         penalization_above_1 = torch.sum(penalization_above_1 - 1)
        
#         penalization = penalization_bellow_0 + penalization_above_1
#         loss["penalization"] = penalization/num_views_per_iteration
        
    
#     #Weighted sum of the losses
#     sum_loss = torch.tensor(0.0, device=device)
#     for k, l in loss.items():
#         sum_loss += l * losses[k]["weight"]
#         losses[k]["values"].append(float(l.detach().cpu()))
     
#     # wandb.log({"loss_rgb": loss["rgb"], "loss_penalization": loss["penalization"], "sum_loss": sum_loss.item()})
    
#     if current_iterations_without_best_loss > iterations_no_best_loss:
#         print(f'Converged at best loss: {best_loss}')
#         # torch.save(origin_texture, store_data + "textures/ground_truth.pt")
#         # with open(store_data + 'ground_truth_final_iterations.txt', 'w') as f:
#         #     f.write(str(i))
#         # break

    

#     # print((sum_loss/ best_loss).item())
#     if (sum_loss <= best_loss*loss_increased_at_least_by).item():
#         # print(f'RESET iterations at: {current_iterations_without_best_loss}')
#         best_loss = sum_loss
#         current_iterations_without_best_loss = 0

    
#     current_iterations_without_best_loss += 1
    
    
#     # Print the losses
#     loop.set_description("total_loss = %.6f" % sum_loss)
    
#     # Plot mesh
#     if i % plot_period == 0:
#         texture_name = 'cow_texture_after_' + str(i) + '_iterations.pt'
#         # torch.save(origin_texture, store_data + 'textures/' + texture_name)
#         visualize_prediction(mesh, renderer=renderer_textured, title="iter: %d" % i, silhouette=False, target_image=predicted_rgb[0])
#         visualize_target(1)
#         visualize_mesh()
#         print(torch.max(origin_texture))
#         print(torch.min(origin_texture))
        
#     # Optimization step
#     sum_loss.backward()
#     my_optimizer.step()




    



# # In[ ]:




