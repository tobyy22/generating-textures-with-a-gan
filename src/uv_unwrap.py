import bpy
import sys
import argparse
import os
import shutil

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Blender UV Unwrap Script")

    # Add arguments
    parser.add_argument("--obj_path", type=str, required=True, help="Path to the OBJ file")
    parser.add_argument("--export_dir", type=str, required=True, help="Directory where output should be saved")
    parser.add_argument("--texture_path", type=str, required=False, help="Path to the texture file")

    # Extract Blender's arguments (everything after --)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    return args


if __name__ == "__main__":

    # Get command-line arguments
    args = parse_args()

    # ✅ Define the output paths
    output_folder = os.path.join(args.export_dir, "placeholder")
    export_path = os.path.join(output_folder, "normalized_model.obj")
    mtl_path = os.path.join(output_folder, "normalized_model.mtl")
    texture_dest = os.path.join(output_folder, "texture.png")  # Standardized texture name

    # ✅ Ensure the export directory exists
    os.makedirs(output_folder, exist_ok=True)
    print(f"Export directory created: {output_folder}")

    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Import the OBJ file
    bpy.ops.wm.obj_import(filepath=args.obj_path)

    # Get the imported object
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    # Ensure the object has an active UV map
    if not obj.data.uv_layers:
        bpy.ops.mesh.uv_texture_add()

    # Apply a new material if a texture is provided
    if args.texture_path:
        # Copy the texture to the export folder with a fixed name (texture.png)
        shutil.copy(args.texture_path, texture_dest)
        print(f"Copied texture to: {texture_dest}")

        # Create a new material and assign it
        mat = bpy.data.materials.new(name="Material")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")

        # Load the copied texture
        tex_image = bpy.data.images.load(texture_dest)
        tex_node = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex_node.image = tex_image

        # Connect the texture node to the BSDF
        mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_node.outputs['Color'])

        # Assign the material to the object
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat

    # Switch to Edit mode to apply UV unwrapping
    bpy.ops.object.mode_set(mode='EDIT')

    # Select all faces
    bpy.ops.mesh.select_all(action='SELECT')

    # Apply Smart UV Project
    bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)

    # Switch back to Object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # ✅ Export the unwrapped OBJ file
    bpy.ops.export_scene.obj(
        filepath=export_path,
        use_selection=True,
        use_materials=True,
        path_mode='RELATIVE'  # Ensures relative paths in MTL file
    )

    # ✅ Fix the MTL file to ensure it references "texture.png"
    with open(mtl_path, "r") as f:
        mtl_content = f.read()

    mtl_content = mtl_content.replace(args.texture_path, "texture.png")  # Force relative reference

    with open(mtl_path, "w") as f:
        f.write(mtl_content)

    print(f"UV unwrapping completed successfully! Exported to {export_path}")
