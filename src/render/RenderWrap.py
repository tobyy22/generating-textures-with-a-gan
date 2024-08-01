#!/usr/bin/env python


import torch
import torchvision
import time

import numpy as np

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes

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

# add path for demo utils functions
import os

import pickle
import random

from src_c.GAN.settings import *


class RenderWrap:
    def __init__(self, image_size, device, max_faces_per_bin):

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
            image_size=image_size,
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
        mesh.textures._maps_padded = texture
        meshes = mesh.extend(self.num_views)
        target_images = self.renderer_textured(
            meshes, cameras=self.cameras, lights=self.lights
        )

        target_rgb = [target_images[i, ..., :3] for i in range(self.num_views)]

        return target_rgb

    def load_mesh_render_with_texture(
        self,
        obj_filename,
        views_to_render,
        camera_indices,
        texture=None,
        non_random=True,
    ):

        """
        Args:
            obj_filename: mesh directory

        Returns:
            mesh: loaded (normalized) mesh
            target_rgb: list of rendered views


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

        mesh.textures._maps_padded = torch.permute(
            mesh.textures._maps_padded, (0, 3, 1, 2)
        )

        mesh.textures._maps_padded = torch.permute(
            mesh.textures._maps_padded, (0, 2, 3, 1)
        )

        # mesh is normalized to sphere of radius 1
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
