#!/usr/bin/env python

import os
import time
import random
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from src.render.RenderWrap import RenderWrap
from src.dataset3d.uv_texture_generator import get_uv_texture


class DataLoader:
    """
    DataLoader class for loading and managing dataset examples with optional invalid data filtering.

    Attributes:
        dataset_directory (str): Path to the dataset directory containing mesh directories.
        number_of_examples (int): Total number of examples to load from the dataset.
        invalid_data_file (str, optional): Path to a file containing invalid data identifiers to be skipped.
    """
    
    def __init__(self, dataset_directory: str, number_of_examples: int, invalid_data_file: str = None):
        """
        Initializes the DataLoader with dataset path, example count, and optional invalid data file.

        Args:
            dataset_directory (str): Directory containing the dataset.
            number_of_examples (int): Number of examples to load.
            invalid_data_file (str, optional): Path to file listing invalid data entries. Defaults to None.
        """
        
        self.dataset_directory = dataset_directory
        self.number_of_examples = number_of_examples
        self.invalid_data_file = invalid_data_file

        # Initialize data containers
        self.invalid_data = []
        self.data_names = []
        self.pre_generated_uv_textures_names = []

        # Load dataset and handle invalid entries
        self.load_data()

        # For indexing and batching
        self.current_index = 0
        self.current_data_batch = []

        # self.data_names = [self.data_names[1433]]
        # print(self.data_names)

    def load_data(self):
        """
        Loads valid data names into the list `self.data_names`. 

        If an `invalid_data_file` is provided, entries listed in it are skipped. This method
        iterates through directories in `dataset_directory`, adding only valid entries to
        `self.data_names`. Each valid entry represents a path to a "normalized_model.obj" file.

        Returns:
            None
        """
        # Load invalid data identifiers, if provided
        if self.invalid_data_file is not None:
            with open(self.invalid_data_file, "rb") as input_file:
                self.invalid_data = pickle.load(input_file)

        # Iterate through dataset directory to gather valid entries
        for dire in os.listdir(self.dataset_directory):
            obj_filename = os.path.join(self.dataset_directory, dire, "normalized_model.obj")
            
            # Skip entries listed in invalid data or hidden directories (those starting with ".")
            if obj_filename in self.invalid_data or dire.startswith("."):
                continue
            
            # Append valid file paths to data_names
            self.data_names.append(obj_filename)
        
        # Sort data names for consistency
        self.data_names = sorted(self.data_names)


    def compute_new_current_data_batch(self):
        """
        Computes and updates the list `self.current_data_batch` with the names of meshes to be rendered
        in a specific iteration. This batch will contain a subset of `data_names`, allowing iteration
        through the dataset in chunks defined by `number_of_examples`.

        The computed `current_data_batch` is used in both `__next__` and `get_fake_data` methods.

        Raises:
            StopIteration: If `current_index` has reached the end of `data_names`, indicating that all 
                           data has been iterated through. Resets `current_index` to 0 upon completion.
        """
        start_index = self.current_index
        end_index = start_index + self.number_of_examples

        # Ensure the end index does not exceed the length of data_names
        if end_index >= len(self.data_names):
            end_index = len(self.data_names)
        
        # If the start index surpasses end index, reset for a new cycle and stop iteration
        if start_index >= end_index:
            self.current_index = 0
            raise StopIteration
        
        # Update the index and current data batch for the next iteration
        self.current_index += self.number_of_examples
        self.current_data_batch = self.data_names[start_index:end_index]

    def get_current_data_batch(self):
        """
        Returns the current data batch.

        Returns:
            list: The list of file paths in `current_data_batch`.
        """
        return self.current_data_batch

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The length of `data_names`.
        """
        return len(self.data_names)

    def __getitem__(self, index):
        """
        Retrieves a data item by its index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            str: The file path at the specified index in `data_names`.
        """
        return self.data_names[index]



class UVTextureDataset(Dataset):
    """
    Iterable Dataset for generating UV textures and loading corresponding target textures.

    Attributes:
        dataloader (DataLoader): Instance of DataLoader to manage dataset entries.
        device (str): Device to be used for generating UV textures.
        image_size (int): Resolution of the generated UV textures (default: 64).
    """

    def __init__(self, dataset_directory, uv_textures_directory, batch_size=16, device='cuda', image_size=64):
        """
        Initializes the UVTextureDataset.

        Args:
            dataloader (DataLoader): Instance of DataLoader to manage dataset entries.
            device (str): Device to be used for generating UV textures.
            image_size (int): Resolution of the generated UV textures. Defaults to 64.
        """
        self.dataloader = DataLoader(
            dataset_directory=dataset_directory,
            number_of_examples=batch_size,
            invalid_data_file="./my_data/nerenderovatelne.txt",
        )        
        self.device = device
        self.uv_textures_directory = uv_textures_directory
        self.image_size = image_size

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.dataloader)

    def __getitem__(self, index):
        """
        Retrieves a data sample by index.

        Args:
            index (int): Index of the data sample to retrieve.

        Returns:
            tuple: A tuple containing the generated UV texture (torch.Tensor) and the target texture (torch.Tensor).
        """
        # Get the path to the "normalized_model.obj" for this index
        obj_filename = self.dataloader[index]
        uuid = os.path.basename(os.path.dirname(obj_filename))
        uv_texture_filename = os.path.join(self.uv_textures_directory, f"{uuid}.pt")
        
        # Load the UV texture
        uv_texture = self.load_uv_texture(uv_texture_filename)

        # Load the target texture
        texture_path = os.path.join(os.path.dirname(obj_filename), "texture.png")
        target_texture = self.load_texture(texture_path)



        return uv_texture, target_texture
    
    def load_uv_texture(self, uv_texture_filename):
        """
        Pre-generated, no preprocessing required.
        """
        uv_texture = torch.load(uv_texture_filename, self.device, weights_only=True)
        uv_texture = torch.squeeze(uv_texture)
        uv_texture = torch.permute(uv_texture, (2, 0, 1))
        uv_texture = uv_texture.to(self.device)

        # Ensure normalization
        min_val = uv_texture.min()
        max_val = uv_texture.max()
        uv_texture = (uv_texture - min_val) / (max_val - min_val)

        return uv_texture



    def load_texture(self, texture_path):
        """
        Loads the target texture as a tensor.

        Args:
            texture_path (str): Path to the target texture file.

        Returns:
            torch.Tensor: Loaded texture as a tensor with shape (C, H, W) normalized to [0, 1].
        """

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor() 
        ])

        image = Image.open(texture_path).convert("RGB")
        texture_tensor = transform(image).to(self.device)
        # texture_tensor = torch.permute(texture_tensor, (2, 0, 1))

        return texture_tensor

    


class Dataset3D:
    """
    This class handles loading and managing a 3D dataset, including rendering meshes with both real and fake
    textures and returning the rendered views as tensors.

    In each iteration:
        - 'number_of_examples' meshes are loaded.
        - For each mesh, 'num_views' different views are generated.
        - Out of these views, only 'number_of_views' are retained per mesh.
    
    Therefore, the final batch size after each iteration will be `number_of_examples * number_of_views`.

    """

    def __init__(
        self,
        dataset_directory,
        number_of_examples,
        number_of_views,
        device,
        invalid_data_file=None,
        pre_generated_uv_textures_dir=None,
        pregenerate_uv_textures=False,
        uv_textures_pregenerated=False,
        image_size=256,
        texture_size=128
    ):
        """
        Initializes the Dataset3D class with parameters for loading, rendering, and managing 3D dataset views.

        Args:
            dataset_directory (str): Directory containing 3D model files.
            number_of_examples (int): Number of examples (meshes) to load per iteration.
            number_of_views (int): Number of random views to generate per mesh in each iteration.
            device (torch.device): Device on which tensors are processed.
            invalid_data_file (str, optional): Path to file listing non-renderable data in the dataset.
            GAN_mode (bool, optional): If True, applies GAN-based rendering techniques. Defaults to True.
            pre_generated_uv_textures_dir (str, optional): Directory containing pre-generated UV textures. Defaults to None.
            pregenerate_uv_textures (bool, optional): If True, generates UV textures in advance. Defaults to False.
            uv_textures_pregenerated (bool, optional): Specifies if UV textures are already pre-generated. Defaults to False.
            image_size (int, optional): Resolution for rendering images. Defaults to 128.
        """

        assert number_of_views <= 20
        self.number_of_views = number_of_views
        self.pre_generated_uv_textures_dir = pre_generated_uv_textures_dir

        # Initialize DataLoader with dataset configuration
        self.dataloader = DataLoader(
            dataset_directory=dataset_directory,
            number_of_examples=number_of_examples,
            invalid_data_file=invalid_data_file,
        )

        self.device = device
        self.camera_indices = None

        # Set up rendering configurations
        self.render_wrapper = RenderWrap(image_size, texture_size, self.device) 
        self.mesh = None
        self.image_size = image_size

        # Texture generation settings
        self.pregenerate_uv_textures = pregenerate_uv_textures
        self.uv_textures_pregenerated = uv_textures_pregenerated

        # Pre-generate UV textures if required
        if pregenerate_uv_textures:
            self.pre_generate_uv_textures()
            self.uv_textures_pregenerated = True

        #for visualization, we will use all 20 views
        self.all_cameras = list(range(20))



    def pre_generate_uv_textures(self):
        """
        Pre-generates UV textures for each mesh in `data_names` if they do not already exist.

        This method saves the generated UV textures to the specified directory (`pre_generated_uv_textures_dir`).
        It uses a concurrent approach with `ThreadPoolExecutor` to speed up the process by generating textures
        in parallel. Only textures that do not already exist are generated.

        Raises:
            AssertionError: If `pre_generated_uv_textures_dir` is None, as the directory must be specified.
        """
        # Ensure the directory for saving pre-generated textures is specified
        assert self.pre_generated_uv_textures_dir is not None, "pre_generated_uv_textures_dir must be specified."

        print(f'Pregenerating UV textures... Texture size: {self.texture_size}')
        os.makedirs(self.pre_generated_uv_textures_dir, exist_ok=True)

        def process_data_name(data_name):
            """
            Processes a single data name to generate and save its UV texture if it does not exist.

            Args:
                data_name (str): The name of the data entry (mesh) for which to generate the UV texture.

            Returns:
                str: A message indicating whether the UV texture was generated or already existed.
            """
            uv_texture_name = self.get_path_to_uv_texture_from_data_name(data_name)
            if not os.path.exists(uv_texture_name):
                # Generate and save UV texture
                uv_texture = get_uv_texture(data_name, self.device, image_size=self.texture_size)
                torch.save(uv_texture, uv_texture_name)
                return f'Generated UV texture: {uv_texture_name}'
            else:
                return f'UV texture already exists: {uv_texture_name}, skipping.'

        # Execute texture generation concurrently
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_data_name, data_name): data_name for data_name in self.dataloader.data_names}
            for future in tqdm(as_completed(futures), total=len(futures)):
                print(future.result())

    def get_path_to_uv_texture_from_data_name(self, data_name):
        """
        Generates the file path for a pre-generated UV texture based on the provided data name.

        Args:
            data_name (str): The name of the data entry (mesh) to be used for generating the file path.

        Returns:
            str: The file path where the UV texture for the given mesh data is or will be stored.
        """
        # Extract mesh ID from the data name and create path to its UV texture file
        mesh_id = data_name.split("/")[-2]
        return os.path.join(self.pre_generated_uv_textures_dir, f"{mesh_id}.pt")


    def load_uv_texture(self, data_name):
        """
        Loads the UV texture for a given data name. If pre-generated UV textures are available, it
        attempts to load the texture from the pre-generated directory. If the pre-generated texture
        is unavailable, it generates a new UV texture.

        Args:
            data_name (str): The name of the data entry (mesh) for which the UV texture is loaded or generated.

        Returns:
            torch.Tensor: The UV texture tensor for the specified data entry.
        """
        if self.uv_textures_pregenerated:
            # Construct path for pre-generated UV texture
            pregenerated_uv_texture_path = self.get_path_to_uv_texture_from_data_name(data_name)
            # Load the texture if it exists in the pre-generated directory
            if os.path.exists(pregenerated_uv_texture_path):
                return torch.load(pregenerated_uv_texture_path, weights_only=True)
        
        # Generate UV texture if pre-generated texture is not available
        return get_uv_texture(data_name, device=self.device, image_size=self.image_size)


    def __iter__(self):
        return self

    def __len__(self):
        """
        Calculates the total number of iterations required to cover the entire dataset in one epoch,
        based on the number of examples loaded per iteration.

        Returns:
            int: The total number of iterations (batches) in one epoch. An additional batch is included
                 if there is a remainder when dividing the dataset length by `number_of_examples`.
        """
        # Determine if an additional batch is needed for remaining examples
        add = 1 if len(self.dataloader) % self.dataloader.number_of_examples != 0 else 0

        # Calculate total iterations by dividing dataset length by batch size and adding any remainder batch
        return len(self.dataloader) // self.dataloader.number_of_examples + add

    def compute_camera_indices(self):
        """
        Computes random camera indices for each example in the dataset, selecting a specified number of views per example.

        This method populates `self.camera_indices` with a list of randomly sampled camera angles for each example
        in the current batch, based on `number_of_views`.

        Returns:
            None
        """
        self.camera_indices = [
            random.sample(range(20), self.number_of_views)
            for _ in range(self.dataloader.number_of_examples)
        ]

    def __next__(self):
        """
        Computes the next batch of rendered views and returns it as a tensor.

        This method:
            - Computes a new batch of data names for rendering.
            - Generates random camera indices for each example.
            - Renders each example with textures based on the specified camera indices.
            - Aggregates and formats the rendered views into a tensor of shape `(batch_size, 3, image_size, image_size)`.

        Returns:
            torch.Tensor: A tensor containing the rendered views. The tensor shape is 
                          `(batch size, 3, image_size, image_size)`, where each view is represented as a 3-channel (RGB) image.
        """
        # Compute a new data batch and associated camera indices
        self.dataloader.compute_new_current_data_batch()
        self.compute_camera_indices()
        
        real_data = []

        # Render each example and collect target RGB views
        for i, batch_elem in enumerate(self.dataloader.get_current_data_batch()):
            target_rgb = self.render_wrapper.load_mesh_render_with_texture(
                batch_elem, camera_indices=self.camera_indices[i]
            )
            real_data.extend(target_rgb)

        # Add an extra dimension to each view and concatenate into a single tensor
        real_data = torch.cat([torch.unsqueeze(view, 0) for view in real_data], dim=0)
        
        # Rearrange dimensions to match the output format (batch_size, 3, image_size, image_size)
        real_data = torch.permute(real_data, (0, 3, 1, 2))

        return real_data

    def render_specific_object(self, index=888):
        """
        Function renders specific object from dataset. Handy for visualization. 
        
        Args:
            index (int): Index of visualized object. 
        
        Returns: 
            torch.Tensor: Tensor containing rendered views of the object. 
        """
        object_name = self.dataloader[index]
        target_rgb = self.render_wrapper.load_mesh_render_with_texture(object_name, self.all_cameras)
        return target_rgb


    def list_of_tensors_to_tensor(self, data):
        """
        Converts list of tensors to tensor. 
        Returns: 
            torch.Tensor
        """
        tensor_from_data = torch.stack(data)
        tensor_from_data = tensor_from_data.permute(0, 3, 1, 2)
        return tensor_from_data
    
    def get_random_data_index(self):
        """
        Function to obtain random data index. Handy when we want to visualize random object to see how the model performs. 
        Returns: 
            int: Random index. 
        """
        dataloder_length = len(self.dataloader)
        return random.randint(0, dataloder_length-1)

    def get_fake_data(self, textures):
        """
        This function will render objects in current batch with provided textures. Intended use is to provide fake textures from the generator to generate fake views. 
        
        Args:
            textures (torch.Tensor): Fake textures to render the objects in current batch with. Textures.size(0) should not be less than self.number_of_examples as
                                     there would not be enough textures.

        Returns: 
            torch.Tensor: Fake views. 

        """

        fake_data = []
        for j, batch_elem in enumerate(self.dataloader.get_current_data_batch()):
            
            permutated_textures = torch.permute(textures, (0, 2, 3, 1))

            fake_rgb = self.render_wrapper.load_mesh_render_with_texture(
                batch_elem,
                camera_indices=self.camera_indices[j],
                texture=torch.unsqueeze(permutated_textures[j], 0),
            )
            fake_data.extend(fake_rgb)

        for k in range(len(fake_data)):
            fake_data[k] = torch.unsqueeze(fake_data[k], 0)

        fake_data = torch.cat(fake_data, 0)
        fake_data = torch.permute(fake_data, (0, 3, 1, 2))

        return fake_data


    def get_fake_data_for_texture_optimization(self, textures):
        """

        Args:
            textures: fake textures to render the meshes with. textures.size(0) should not be less than self.number_of_examples

        Returns: Tensor of fake views. The tensor size will is (batch size, 3, image_size, image_size).

        """

        fake_data = []
        for j, _ in enumerate(self.dataloader.get_current_data_batch()):

            permutated_textures = textures

            fake_rgb = self.render_wrapper.load_mesh_render_with_texture(
                None,
                camera_indices=self.camera_indices[j],
                texture=torch.unsqueeze(permutated_textures[j], 0),
            )

            fake_data.extend(fake_rgb)

        for k in range(len(fake_data)):
            fake_data[k] = torch.unsqueeze(fake_data[k], 0)

        fake_data = torch.cat(fake_data, 0)
        fake_data = torch.permute(fake_data, (0, 3, 1, 2))

        return fake_data

    def load_current_batch_position_textures(self):
        """
        Function loads position textures for objects in current batch. 

        Returns: 
            torch.Tensor: Position textures for current batch. 
        """

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
        if texture is not None:
            texture = torch.permute(texture, (0, 2, 3, 1))
        views = self.render_wrapper.load_mesh_render_with_texture(
            obj_filename, self.all_cameras, texture=texture
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


def render_object_with_texture(object_path, texture, cameras=list(range(20))):
    """
    This function will render objects in current batch with provided textures. Intended use is to provide fake textures from the generator to generate fake views. 
    
    Args:
        textures (torch.Tensor): Fake textures to render the objects in current batch with. Textures.size(0) should not be less than self.number_of_examples as
                                    there would not be enough textures. 

    Returns: 
        torch.Tensor: Fake views. 

    """


    permutated_textures = torch.permute(texture, (0, 2, 3, 1))
    fake_rgb = self.render_wrapper.load_mesh_render_with_texture(
        object_path,
        camera_indices=cameras,
        texture=torch.unsqueeze(permutated_textures[0], 0),
    )


    return fake_rgb