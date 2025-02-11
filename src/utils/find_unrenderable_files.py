import pickle
import os
from PIL import Image

def update_unrenderable_files(inv_file_path, source_dir, img_threshold=2000):
    """
    Updates a list of unrenderable files based on image dimensions in the specified directory.

    Args:
    - inv_file_path (str): Path to the pickle file storing the list of unrenderable files.
    - source_dir (str): Directory containing the model subdirectories with textures.
    - img_threshold (int): The threshold for image dimensions to consider a file unrenderable.

    Returns:
    - None
    """
    # Load the existing list of unrenderable files
    try:
        with open(inv_file_path, "rb") as input_file:
            inv = pickle.load(input_file)
    except (FileNotFoundError, EOFError):
        inv = []

    # Iterate over each subdirectory in the source directory
    for dire in os.listdir(source_dir):
        obj_filename = os.path.join(source_dir, dire, 'texture.png')
        try:
            img = Image.open(obj_filename)
            file_to_ignore = os.path.join(source_dir, dire, 'normalized_model.obj')
            width, height = img.size

            # Check if the image exceeds the threshold and if the file is not already in the list
            if (width > img_threshold or height > img_threshold) and file_to_ignore not in inv:
                inv.append(file_to_ignore)
                print(f"Added {file_to_ignore} to ignore list due to large size: {width}x{height}")
        
        except Exception as e: 
            print(f"Error processing {obj_filename}: {e}")
            continue

    # Save the updated list back to the pickle file
    with open(inv_file_path, "wb") as output_file:
        pickle.dump(inv, output_file)

# Example usage
inv_file_path = 'nerenderovatelne.txt'
source_dir = '/projects/3DDatasets/3D-FUTURE/3D-FUTURE-model'
update_unrenderable_files(inv_file_path, source_dir)
