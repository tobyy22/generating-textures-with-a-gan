#!/usr/bin/env python


import torch
import torchvision
import torch.nn.functional as F

import time
from PIL import Image


import numpy as np

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
import torchvision.transforms as transforms


# Data structures and functions for rendering
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams
)



"""
RenderWrap: A class for handling 3D mesh rendering using PyTorch3D.

This class provides utilities for rendering 3D meshes with textures from multiple views using PyTorch3D.
It is designed to handle various rendering settings, including multiple camera views, lighting, and rasterization settings.

"""

class RenderWrap:
    def __init__(self, rendered_image_size, texture_size, device, max_faces_per_bin=310000):
        """
        Initialize the RenderWrap class with rendering settings.

        Args:
            rendered_image_size (int): The size of the rendered image (height and width).
            texture_size (tuple): The size of the texture (width, height).
            device (str): The device to use for rendering ("cuda" or "cpu").
            max_faces_per_bin (int): Maximum number of faces per bin for efficient rasterization.
        """
        self.rendered_image_size = rendered_image_size
        self.texture_size = texture_size
        self.device = device
        self.num_views = 20
        self.last_mesh = None

        elev = torch.linspace(0, 360, self.num_views)
        azim = torch.linspace(-180, 180, self.num_views)
        R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        self.cameras = OpenGLPerspectiveCameras(device=self.device, R=R, T=T)
        self.camera = OpenGLPerspectiveCameras(
            device=self.device, R=R[None, 1, ...], T=T[None, 1, ...]
        )

        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size=rendered_image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
            faces_per_pixel=50,
            perspective_correct=False,
            max_faces_per_bin=max_faces_per_bin,
        )

        blend_params = BlendParams(1e-4, 1e-4, (1, 1, 1))
        self.renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera, raster_settings=raster_settings_soft
            ),
            shader=SoftPhongShader(
                device=self.device, cameras=self.camera, lights=self.lights, blend_params=blend_params
            ),
        )

    def render_mesh_with_texture(self, mesh, texture):
        """
        Render a 3D mesh with the provided texture.

        Args:
            mesh (Mesh): The 3D mesh object to render.
            texture (Tensor): The texture to apply to the mesh.

        Returns:
            List[Tensor]: A list of rendered views as images.
        """
        mesh.textures._maps_padded = texture
        meshes = mesh.extend(self.num_views)
        target_images = self.renderer_textured(
            meshes, cameras=self.cameras, lights=self.lights
        )

        target_rgb = [target_images[i, ..., :3] for i in range(self.num_views)]
        return target_rgb

    def load_mesh_render_with_texture(self, obj_filename, camera_indices, texture=None, non_random=True):
        """
        Load a mesh from a file and render it with the provided texture from multiple views.

        Args:
            obj_filename (str): Path to the .obj file containing the 3D mesh.
            camera_indices (List[int]): Indices of the camera views to render from.
            texture (Tensor, optional): The texture to apply to the mesh. Defaults to None.
            non_random (bool, optional): Whether to use deterministic camera indices. Defaults to True.

        Returns:
            List[Tensor]: Rendered views as RGB images.
        """
        if texture is not None:
            assert texture.shape[0] == 1
        mesh = None
        if obj_filename is None:
            mesh = self.last_mesh
        else:
            mesh = load_objs_as_meshes([obj_filename], device=self.device)
            self.last_mesh = mesh

        if texture is not None:
            mesh.textures._maps_padded = texture

        # Normalize and downscale the texture
        mesh.textures._maps_padded = mesh.textures._maps_padded.permute(0, 3, 1, 2)
        mesh.textures._maps_padded = F.interpolate(
            mesh.textures._maps_padded, size=(self.texture_size, self.texture_size), mode="bilinear", align_corners=False
        )
        mesh.textures._maps_padded = mesh.textures._maps_padded.permute(0, 2, 3, 1)

        # Normalize the mesh to a sphere of radius 1
        verts = mesh.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))

        random_cameras = self.cameras[camera_indices]
        meshes = mesh.extend(len(camera_indices))
        target_images = self.renderer_textured(
            meshes, cameras=random_cameras, lights=self.lights
        )

        target_rgb = [target_images[i, ..., :3] for i in range(len(camera_indices))]

        return target_rgb
