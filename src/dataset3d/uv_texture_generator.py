#!/usr/bin/env python



import torch
import pytorch3d
import pytorch3d.io



def get_uv_texture(obj_filename, device, image_size=128):
    """
    Generates a UV texture from a 3D object file (.obj) and returns it as a torch tensor.

    This function loads a 3D object file (OBJ format), extracts its UV coordinates, and
    generates a texture by mapping vertex positions or normals to colors. The resulting
    UV texture is rendered using PyTorch3D's rasterizer and returned as a torch tensor.

    Args:
        obj_filename (str): The path to the .obj file containing the 3D object.
        device (torch.device): The PyTorch device (CPU or GPU) on which computations will be performed.
        image_size (int, optional): The resolution of the output texture image (default: 1024).

    Returns:
        torch.Tensor: A rendered UV texture image of shape (image_size, image_size, 3),
        representing the RGB color mapping of the object's UV coordinates.
    """

    obj_mesh = pytorch3d.io.load_objs_as_meshes([obj_filename], device=device)
    
    # create a fake 3D object - just triangles in a plane
    #  - positions: (X, Y) == (U, V)
    #  - colors: (R, G, B) == (X, Y, Z) or (R, G, B) == normals
    triangle_verts = []
    triangle_faces = []
    vertex_colors = []
    assert len(obj_mesh.faces_list()) == 1
    assert len(obj_mesh.verts_list()) == 1
    assert len(obj_mesh.textures.verts_uvs_list()) == 1
    assert len(obj_mesh.textures.faces_uvs_list()) == 1
    assert len(obj_mesh.faces_normals_list()) == 1
    faces = obj_mesh.faces_list()[0]
    face_uv_idx = obj_mesh.textures.faces_uvs_list()[0]
    uvs = obj_mesh.textures.verts_uvs_list()[0]

    verts = obj_mesh.verts_list()[0]
    center = torch.min(verts, dim=0).values + torch.max(verts, dim=0).values
    center = center.to(device)
    verts -= center
    verts = verts/verts.abs().max() / 2 + .5 # normalize to 0..1

    normals = obj_mesh.verts_normals_list()[0] / 2 + .5 # -1..1 to 0..1

    for fi, f in enumerate(faces):
        assert f.shape == torch.Size([3]) # triangle (vertex indices)
        assert uvs[face_uv_idx[fi]].shape == torch.Size([3, 2]) # UV coords (2 values)
        
        # generate vertex position according to the UV (z := 0)
        triangle_verts.append(torch.nn.ConstantPad1d((0, 1, 0, 0), 0.)(uvs[face_uv_idx[fi]]))

        # generate vertex indices
        triangle_faces.append([0 + fi * 3, 1 + fi * 3, 2 + fi * 3])
        
        # generate vertex colors (based on the original vertex position, or normal) - ! switch here !
        vertex_colors.append(verts[f])
        #vertex_colors.append(normals[f])
        # TODO more features like curvature, triangle area etc.

    triangle_verts = torch.cat(triangle_verts)
    triangle_faces = torch.Tensor(triangle_faces).to(device)
    vertex_colors = torch.cat(vertex_colors)



    mesh = pytorch3d.structures.Meshes(
        verts=[triangle_verts*2-1],
        faces=[triangle_faces],
        textures=pytorch3d.renderer.Textures(verts_rgb=[vertex_colors])
    )


    # render/rasterize the fake 3D object (no lighting)
    R, T = pytorch3d.renderer.look_at_view_transform(eye=torch.Tensor([0,0,1]).reshape((-1,3)))
    camera = pytorch3d.renderer.cameras.OrthographicCameras(R=R, T=T, device=device)
    renderer = pytorch3d.renderer.MeshRenderer(
        rasterizer=pytorch3d.renderer.MeshRasterizer(
            cameras=camera, 
            raster_settings=pytorch3d.renderer.RasterizationSettings(
                image_size=image_size, 
                faces_per_pixel=1, 
                perspective_correct=False,
            )
        ),
    #    shader=pytorch3d.renderer.mesh.shader.HardPhongShader( # computes colors per-pixel
        shader=pytorch3d.renderer.mesh.shader.HardGouraudShader( # enough if we only interpolate values from vertices
            device=device, 
            cameras=camera,
            lights=pytorch3d.renderer.lighting.AmbientLights(device=device)
        )
    )
    img = renderer(mesh)[..., :3] # just RGB

    return img
