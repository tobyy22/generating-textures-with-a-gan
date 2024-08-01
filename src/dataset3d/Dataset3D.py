#!/usr/bin/env python


import torch
from tqdm import tqdm

import numpy as np

import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# Util function for loading meshes


# add path for demo utils functions
import os

import pickle
import random

from src_c.render.RenderWrap import RenderWrap
from src_c.dataset3d.uv_texture_generator import get_uv_texture
from src_c.GAN.settings import *


class DataLoader:
    def __init__(
        self,
        dataset_directory,
        number_of_examples,
        invalid_data_file=None,
    ):


        self.dataset_directory = dataset_directory
        self.number_of_examples = number_of_examples
        self.invalid_data_file = invalid_data_file

        # all data names (names of directories with meshes) will be stored in data_names
        # invalid_data_list will be skipped
        self.invalid_data = []
        self.data_names = []
        self.pre_generated_uv_textures_names = []
        self.load_data()

        # indexing
        self.current_index = 0
        self.current_data_batch = []

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
            obj_filename = os.path.join(
                self.dataset_directory, dire, "normalized_model.obj"
            )
            if obj_filename in self.invalid_data:
                continue
            if dire[0] == ".":
                continue
            self.data_names.append(obj_filename)
            # if self.pre_generated_uv_textures_dir:
            #     self.pre_generated_uv_textures_names.append(
            #         os.path.join(self.pre_generated_uv_textures_dir, dire + ".pt")
            #     )
        self.data_names = sorted(self.data_names)


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
            # random.shuffle(self.data_names)
            raise StopIteration
        self.current_index += self.number_of_examples
        self.current_data_batch = self.data_names[start_index:end_index]

    def get_current_data_batch(self):
        return self.current_data_batch

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        return self.data_names[index]


class Dataset3D:

    """
    This class takes care of loading the dataset. This class will load meshes, render them both with real and fake
    textures and return the rendered views as tensors.

    In each iteration, it will load 'number_of_examples' meshes. After rendering, we will get 'num_views' views of each mesh.
    So eventually we would have a batch of size number_of_examples*num_views. But from num_views of each mesh, we will only
    keep number_of_views.

    So the final size of the batch that is returned after each iteration is number_of_examples*number_of_views.
    """

    def __init__(
        self,
        dataset_directory,
        number_of_examples,
        number_of_views,
        device,
        invalid_data_file=None,
        GAN_mode=True,
        pre_generated_uv_textures_dir=None,
        pregenerate_uv_textures=False,
        uv_textures_pregenerated=False,
        image_size=128,

    ):

        """
        Args:
            dataset_directory: directory with 3D models
            number_of_examples: number of examples to load each iteration
            number_of_views: number of random views on each mesh each iteration
            invalid_data_file: some data from the dataset might not be renderable
        """

        

        self.number_of_views = number_of_views
        self.pre_generated_uv_textures_dir = pre_generated_uv_textures_dir


        self.dataloader = DataLoader(
            dataset_directory=dataset_directory,
            number_of_examples=number_of_examples,
            invalid_data_file=invalid_data_file,
        )

        self.GAN_mode = GAN_mode
        self.camera_incides = None


        self.device = device

        self.render_wrapper = RenderWrap(image_size, self.device, 310000)
        self.mesh = None
        self.image_size=image_size

        self.pregenerate_uv_textures = pregenerate_uv_textures
        self.uv_textures_pregenerated = uv_textures_pregenerated

        if pregenerate_uv_textures:
            self.pre_generate_uv_textures()
            self.uv_textures_pregenerated = True


    def pre_generate_uv_textures(self):
        assert self.pre_generated_uv_textures_dir is not None

        print(f'Pregenerating uv textures...image size: {self.image_size}')
        os.makedirs(self.pre_generated_uv_textures_dir, exist_ok=True)

        def process_data_name(data_name):
            uv_texture_name = self.get_path_to_uv_texture_from_data_name(data_name)
            if not os.path.exists(uv_texture_name):
                uv_texture = get_uv_texture(data_name, self.device, image_size=self.image_size)
                torch.save(uv_texture, uv_texture_name)
                return f'Generated UV texture: {uv_texture_name}'
            else:
                return f'UV texture already exists: {uv_texture_name}, skipping.'

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_data_name, data_name): data_name for data_name in self.dataloader.data_names}
            for future in tqdm(as_completed(futures), total=len(futures)):
                print(future.result())

    def get_path_to_uv_texture_from_data_name(self, data_name):
        mesh_id = data_name.split("/")[-2]
        return os.path.join(self.pre_generated_uv_textures_dir, mesh_id + ".pt")

    def load_uv_texture(self, data_name):
        if self.uv_textures_pregenerated:
            pregenerated_uv_texture_path = self.get_path_to_uv_texture_from_data_name(data_name)
            if os.path.exists(pregenerated_uv_texture_path):
                return torch.load(pregenerated_uv_texture_path)
        return get_uv_texture(data_name, device=self.device, image_size=self.image_size)


    def __iter__(self):
        return self

    def __len__(self):
        """
        Returns: Length of dataset – how many iterations will be performed in each epoch.

        """
        add = 0
        if len(self.dataloader) % self.dataloader.number_of_examples != 0:
            add = 1

        return len(self.dataloader) // self.dataloader.number_of_examples + add

    def compute_camera_indices(self):
        self.camera_incides = []
        for i in range(self.dataloader.number_of_examples):
            if self.GAN_mode:
                self.camera_incides.append(
                    sorted(random.sample(range(20), self.number_of_views))
                )
            else:
                self.camera_incides.append(
                    random.sample(range(20), self.number_of_views)
                )

    def __next__(self):
        """

        Returns: Tensor of real views. The tensor size will is (batch size, 3, image_size, image_size).

        """
        self.dataloader.compute_new_current_data_batch()
        self.compute_camera_indices()
        real_data = []
        for i, batch_elem in enumerate(self.dataloader.get_current_data_batch()):
            target_rgb = self.render_wrapper.load_mesh_render_with_texture(
                batch_elem, self.number_of_views, camera_indices=self.camera_incides[i]
            )
            real_data.extend(target_rgb)

        for k in range(len(real_data)):
            real_data[k] = torch.unsqueeze(real_data[k], 0)

        real_data = torch.cat(real_data, 0)
        real_data = torch.permute(real_data, (0, 3, 1, 2))

        return real_data

    def render_specific_object(self, index):
        object_name = self.dataloader[index]
        cameras=list(range(20))
        target_rgb = self.render_wrapper.load_mesh_render_with_texture(object_name, self.number_of_views, cameras)
        return target_rgb


    def list_of_tensors_to_tensor(self, data):
        tensor_from_data = torch.stack(data)
        tensor_from_data = tensor_from_data.permute(0, 3, 1, 2)
        return tensor_from_data
    
    def get_random_data_index(self):
        dataloder_length = len(self.dataloader)
        return random.randint(0, dataloder_length-1)

    # TODO: rename to generate_fake_renders
    def get_fake_data(self, textures):
        """

        Args:
            textures: fake textures to render the meshes with. textures.size(0) should not be less than self.number_of_examples

        Returns: Tensor of fake views. The tensor size will is (batch size, 3, image_size, image_size).

        """

        fake_data = []
        for j, batch_elem in enumerate(self.dataloader.get_current_data_batch()):

            permutated_textures = torch.permute(textures, (0, 2, 3, 1))
            fake_rgb = self.render_wrapper.load_mesh_render_with_texture(
                batch_elem,
                self.number_of_views,
                camera_indices=self.camera_incides[j],
                texture=torch.unsqueeze(permutated_textures[j], 0),
            )
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
        for j, batch_elem in enumerate(self.dataloader.get_current_data_batch()):

            permutated_textures = textures

            fake_rgb = self.render_wrapper.load_mesh_render_with_texture(
                None,
                self.number_of_views,
                camera_indices=self.camera_incides[j],
                texture=torch.unsqueeze(permutated_textures[j], 0),
            )

            fake_data.extend(fake_rgb)

        for k in range(len(fake_data)):
            fake_data[k] = torch.unsqueeze(fake_data[k], 0)

        fake_data = torch.cat(fake_data, 0)
        fake_data = torch.permute(fake_data, (0, 3, 1, 2))

        return fake_data

    def load_current_batch_position_textures(self):
        position_textures = []
        for j, batch_elem in enumerate(self.dataloader.get_current_data_batch()):
            texture = self.load_uv_texture(batch_elem)
            position_textures.extend(texture)

        for k in range(len(position_textures)):
            position_textures[k] = torch.unsqueeze(position_textures[k], 0)

        position_textures = torch.cat(position_textures, 0)
        position_textures = torch.permute(position_textures, (0, 3, 1, 2))

        return position_textures

    def load_specific_position_textures(self, index=1290):
        position_textures = []
        texture = self.load_uv_texture(self.dataloader[index])
        position_textures.extend(texture)

        for k in range(len(position_textures)):
            position_textures[k] = torch.unsqueeze(position_textures[k], 0)

        position_textures = torch.cat(position_textures, 0)
        position_textures = torch.permute(position_textures, (0, 3, 1, 2))

        return position_textures

    def get_multiple_views(self, index, texture=None):
        obj_filename = self.dataloader[index]
        cameras = list(range(20))
        if texture is not None:
            texture = torch.permute(texture, (0, 2, 3, 1))
        views = self.render_wrapper.load_mesh_render_with_texture(
            obj_filename, None, cameras, texture=texture
        )
        for k in range(len(views)):
            views[k] = torch.unsqueeze(views[k], 0)

        views = torch.cat(views, 0)

        views = torch.permute(views, (0, 3, 1, 2))
        return views

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
            mesh_name = self.dataloader[i]
        else:
            mesh_name = random.choice(self.dataloader.get_current_data_batch())

        permutated_textures = torch.permute(textures, (0, 2, 3, 1))
        number_of_textures = textures.size(0)

        fake_data = []
        for i in range(number_of_textures):
            fake_rgb = self.render_wrapper.load_mesh_render_with_texture(
                mesh_name,
                self.number_of_views,
                camera_indices=[0],
                texture=torch.unsqueeze(permutated_textures[i], 0),
            )
            fake_rgb = fake_rgb[0]
            fake_rgb = (fake_rgb - torch.min(fake_rgb)) / (
                torch.max(fake_rgb) - torch.min(fake_rgb)
            )
            fake_data.append(fake_rgb)

        for k in range(len(fake_data)):
            fake_data[k] = torch.unsqueeze(fake_data[k], 0)

        fake_data = torch.cat(fake_data, 0)

        fake_data = torch.permute(fake_data, (0, 3, 1, 2))

        return fake_data
