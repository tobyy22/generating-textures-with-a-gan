#!/usr/bin/env python


import torch
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

)

# add path for demo utils functions 
import os

import pickle
import random


class RenderWrap:

    def __init__(self, image_size):

        if torch.cuda.is_available():
            self.device = torch.device("cuda:1")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")

        self.num_views = 20

        self.last_mesh = None

        elev = torch.linspace(0, 360, self.num_views)
        azim = torch.linspace(-180, 180, self.num_views)
        R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        self.cameras = OpenGLPerspectiveCameras(device=self.device, R=R, T=T)
        self.camera = OpenGLPerspectiveCameras(device=self.device, R=R[None, 1, ...],
                                        T=T[None, 1, ...])

        sigma = 1e-4
        raster_settings_soft = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
            faces_per_pixel=50,
            perspective_correct=False,
            max_faces_per_bin=100000
        )

        self.renderer_textured = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera,
                raster_settings=raster_settings_soft
            ),
            shader=SoftPhongShader(device=self.device,
                                cameras=self.camera,
                                lights=self.lights)
        )

    def render_mesh_with_texture(self, mesh, texture):
        mesh.textures._maps_padded = texture
        meshes = mesh.extend(self.num_views)
        target_images = self.renderer_textured(meshes, cameras=self.cameras, lights=self.lights)

        target_rgb = [target_images[i, ..., :3] for i in range(self.num_views)]

        return target_rgb

    def load_mesh_render_with_texture(self, obj_filename, views_to_render,camera_indices, texture=None, non_random=True):

        """
        Args:
            obj_filename: mesh directory

        Returns:
            mesh: loaded (normalized) mesh
            target_rgb: list of rendered views


        """
        mesh = None
        if obj_filename is None:
            mesh = self.last_mesh
        else:
            mesh = load_objs_as_meshes([obj_filename], device=self.device)
            self.last_mesh = mesh

        if texture is not None: 
            mesh.textures._maps_padded = texture

        #mesh is normalized to sphere of radius 1
        verts = mesh.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center)
        mesh.scale_verts_((1.0 / float(scale)))
        
        
        random_cameras = self.cameras[camera_indices]

        # if non_random:
        #     random_cameras = self.cameras
        
        # else:
        #     if views_to_render == -1:
        #         random_cameras = self.camera
        #         views_to_render = 1
        #     else:
        #         random_cameras = self.cameras[random.sample(range(self.num_views), views_to_render)]


        
        meshes = mesh.extend(len(camera_indices))

        # print(f'rendertime: {time.time() - start}')




        #rychlost ok
        target_images = self.renderer_textured(meshes, cameras=random_cameras, lights=self.lights)




        target_rgb = [target_images[i, ..., :3] for i in range(len(camera_indices))]




        return target_rgb
