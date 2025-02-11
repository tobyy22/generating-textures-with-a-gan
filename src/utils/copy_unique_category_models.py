import json
import shutil
import os

def copy_unique_category_models(json_file_path, source_dir, dest_dir, category = 'category'):
    """
    Copies directories of models from the source directory to the destination directory
    based on unique categories found in the provided JSON file.
    
    Args:
    - json_file_path (str): Path to the JSON file containing model information.
    - source_dir (str): Directory where the model folders are located.
    - dest_dir (str): Directory where the model folders will be copied to.
    - category (str): Can also be 'super-category'.
    
    Returns:
    - None
    """
    # Open and load the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Initialize a set to track unique categories
    categories = set()

    # Iterate over each item in the JSON data
    for item in data:
        category = item[category]
        model_id = item["model_id"]
        
        # If the category is unique, copy the model directory
        if category not in categories:
            categories.add(category)
            source_path = os.path.join(source_dir, model_id)
            dest_path = os.path.join(dest_dir, model_id)
            
            # Copy the model directory to the destination
            shutil.copytree(source_path, dest_path)
            print(f"Copied {source_path} to {dest_path}")


if __name__ == '__main__':
    # Example usage
    json_file_path = '/projects/3DDatasets/3D-FUTURE/3D-FUTURE-model/model_info.json'
    source_dir = '/projects/3DDatasets/3D-FUTURE/3D-FUTURE-model/'
    dest_dir = ''

    copy_unique_category_models(json_file_path, source_dir, dest_dir)
