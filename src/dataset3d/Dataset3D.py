#!/usr/bin/env python


import torch

import numpy as np

import time

# Util function for loading meshes


# add path for demo utils functions 
import os

import pickle
import random

from render.RenderWrap import RenderWrap



class Dataset3D:

    """
    This class takes care of loading the dataset. This class will load meshes, render them both with real and fake
    textures and return the rendered views as tensors.

    In each iteration, it will load 'number_of_examples' meshes. After rendering, we will get 'num_views' views of each mesh.
    So eventually we would have a batch of size number_of_examples*num_views. But from num_views of each mesh, we will only
    keep number_of_views.

    So the final size of the batch that is returned after each iteration is number_of_examples*number_of_views.
    """

    def __init__(self, dataset_directory, number_of_examples, number_of_views, invalid_data_file=None, GAN_mode=True):

        """
        Args:
            dataset_directory: directory with 3D models
            number_of_examples: number of examples to load each iteration
            number_of_views: number of random views on each mesh each iteration
            invalid_data_file: some data from the dataset might not be renderable
        """
        self.dataset_directory = dataset_directory
        self.number_of_examples = number_of_examples
        self.number_of_views = number_of_views
        self.invalid_data_file = invalid_data_file

        self.GAN_mode = GAN_mode
        self.camera_incides = None

        #all data names (names of directories with meshes) will be stored in data_names
        #invalid_data_list will be skipped
        self.invalid_data = []
        self.data_names = []
        self.load_data()

        #indexing
        self.current_index = 0
        self.current_data_batch = []

        self.render_wrapper = RenderWrap(64)
        self.mesh = None

    def load_data(self):
        """

        Function to load data names into list self.data_names. If invalid_data_file provided,
        those examples will be skipped.

        Returns: None

        """
        if self.invalid_data_file is not None:
            with open(self.invalid_data_file, "rb") as input_file:
                self.invalid_data = pickle.load(input_file)

        for dire in os.listdir(self.dataset_directory):
            obj_filename = os.path.join(self.dataset_directory, dire, 'normalized_model.obj')
            if obj_filename in self.invalid_data:
                continue
            if dire[0] == '.':
                continue
            self.data_names.append(obj_filename)
        self.data_names = sorted(self.data_names)


    def __iter__(self):
        return self

    def __len__(self):
        """
        Returns: Length of dataset – how many iterations will be performed in each epoch.

        """
        add = 0
        if len(self.data_names) % self.number_of_examples != 0:
            add = 1

        return len(self.data_names) // self.number_of_examples + add

    def compute_new_current_data_batch(self):
        """

        Computes the list self.current_data_batch. This will be the names of all meshes that will be rendered
        in one specific iteration.

        This list is used in both functions __next__ and get_fake_data.

        """
        end_index = self.current_index + self.number_of_examples
        start_index = self.current_index
        if end_index >= (len(self.data_names)):
            end_index = len(self.data_names)
        if start_index >= end_index:
            self.current_index = 0
            raise StopIteration
        self.current_index += self.number_of_examples
        self.current_data_batch = self.data_names[start_index:end_index]

    def compute_camera_indices(self):
        self.camera_incides = []
        for i in range(self.number_of_examples):
            if self.GAN_mode:
                self.camera_incides.append(sorted(random.sample(range(20), self.number_of_views)))
            else:
                self.camera_incides.append(random.sample(range(20), self.number_of_views))

    def __next__(self):
        """

        Returns: Tensor of real views. The tensor size will is (batch size, 3, image_size, image_size).

        """
        self.compute_new_current_data_batch()
        self.compute_camera_indices()
        real_data = []
        for i, batch_elem in enumerate(self.current_data_batch):
            target_rgb = self.render_wrapper.load_mesh_render_with_texture(batch_elem, self.number_of_views, camera_indices=self.camera_incides[i])
            real_data.extend(target_rgb)

        for k in range(len(real_data)):
            real_data[k] = torch.unsqueeze(real_data[k], 0)

        real_data = torch.cat(real_data, 0)
        real_data = torch.permute(real_data, (0, 3, 1, 2))

        return real_data

    def get_fake_data(self, textures):
        """

        Args:
            textures: fake textures to render the meshes with. textures.size(0) should not be less than self.number_of_examples

        Returns: Tensor of fake views. The tensor size will is (batch size, 3, image_size, image_size).

        """


        fake_data = []
        for j, batch_elem in enumerate(self.current_data_batch):

            permutated_textures = torch.permute(textures, (0, 2, 3, 1))

            fake_rgb = self.render_wrapper.load_mesh_render_with_texture(batch_elem, self.number_of_views, camera_indices=self.camera_incides[j], texture=torch.unsqueeze(permutated_textures[j], 0))


            fake_data.extend(fake_rgb)

        for k in range(len(fake_data)):
            fake_data[k] = torch.unsqueeze(fake_data[k], 0)

        fake_data = torch.cat(fake_data, 0)
        fake_data = torch.permute(fake_data, (0, 3, 1, 2))

        return fake_data


    def get_fake_data2(self, textures):
        """

        Args:
            textures: fake textures to render the meshes with. textures.size(0) should not be less than self.number_of_examples

        Returns: Tensor of fake views. The tensor size will is (batch size, 3, image_size, image_size).

        """

        fake_data = []
        for j, batch_elem in enumerate(self.current_data_batch):

            permutated_textures = textures

            fake_rgb = self.render_wrapper.load_mesh_render_with_texture(None, self.number_of_views, camera_indices=self.camera_incides[j], texture=torch.unsqueeze(permutated_textures[j], 0))

            fake_data.extend(fake_rgb)

        for k in range(len(fake_data)):
            fake_data[k] = torch.unsqueeze(fake_data[k], 0)

        fake_data = torch.cat(fake_data, 0)
        fake_data = torch.permute(fake_data, (0, 3, 1, 2))


        return fake_data

    def texture_random_mesh_with_different_textures(self, textures, i=None):
        """
        Loads one mesh at index i. If i is not specified then random mesh is chosen. Function will render this mesh with fake textu
        Args:
            textures:
            i:

        Returns:

        """
        mesh_name = None
        if i is not None:
            mesh_name = self.data_names[i]
        else:
            mesh_name = random.choice(self.data_names)

        permutated_textures = torch.permute(textures, (0, 2, 3, 1))
        number_of_textures = textures.size(0)

        fake_data = []
        for i in range(number_of_textures):
            fake_rgb = self.render_wrapper.load_mesh_render_with_texture(mesh_name, self.number_of_views, camera_indices=[0], texture=torch.unsqueeze(permutated_textures[i], 0))
            fake_rgb = fake_rgb[0]
            fake_rgb = (fake_rgb - torch.min(fake_rgb)) / (torch.max(fake_rgb) - torch.min(fake_rgb))
            fake_data.append(fake_rgb)
        
        for k in range(len(fake_data)):
            fake_data[k] = torch.unsqueeze(fake_data[k], 0)

        fake_data = torch.cat(fake_data, 0)

        fake_data = torch.permute(fake_data, (0, 3, 1, 2))

        return fake_data
