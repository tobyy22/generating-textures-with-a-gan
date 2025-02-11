import warnings
from src.render.RenderWrap import RenderWrap
from src.dataset3d.Dataset3D import DataLoader

def find_optimal_max_faces_per_bin(dataset, initial_max_faces_per_bin=1000, image_size=64, device='cuda:0'):
    """
    Finds the optimal max_faces_per_bin value that avoids rendering warnings.

    Args:
    - dataset (DataLoader): The dataset object containing 3D models.
    - initial_max_faces_per_bin (int): The initial value for max_faces_per_bin.
    - image_size (int): The size of the rendered image.
    - device (str): The device to use for rendering (e.g., 'cuda:0').

    Returns:
    - int: The optimal max_faces_per_bin value.
    """
    max_faces_per_bin = initial_max_faces_per_bin

    def render_object(obj, max_faces_per_bin):
        """
        Renders an object and checks if it triggers any warnings.

        Args:
        - obj: The 3D object to render.
        - max_faces_per_bin (int): The current value of max_faces_per_bin.

        Returns:
        - bool: True if a warning was raised during rendering, False otherwise.
        """
        with warnings.catch_warnings(record=True) as warns:
            renderer = RenderWrap(image_size=image_size, device=device, max_faces_per_bin=max_faces_per_bin)
            images = renderer(meshes_world=obj, R=R, T=T)
            return len(warns) > 0

    while True:
        num_problematic = 0
        for obj in dataset:
            if render_object(obj, max_faces_per_bin):
                num_problematic += 1

        if num_problematic == 0:
            return max_faces_per_bin
        else:
            max_faces_per_bin *= 2

if __name__ == "__main__":
    # Load the dataset
    data = DataLoader(
        dataset_directory="/projects/3DDatasets/3D-FUTURE/3D-FUTURE-model", 
        number_of_examples=1, 
        invalid_data_file="./my_data/nerenderovatelne.txt"
    )

    # Find the optimal max_faces_per_bin value
    optimal_max_faces_per_bin = find_optimal_max_faces_per_bin(data)
    
    print(f"Optimal max_faces_per_bin: {optimal_max_faces_per_bin}")
