import os
import shutil

from torchvision.utils import save_image

    
def normalize_tensor(tensor, normalize=True):
    if normalize:
        tensor = tensor.clone()
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor

def save_tensors_to_png(tensors, path):
    """
    Save a list of tensors to PNG files in a specified path.

    Parameters:
    tensors (list of torch.Tensor): List of tensors with shape (128, 128, 3).
    path (str): Directory where the PNG files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)
    
    # Iterate over the tensors and save each one
    for i, tensor in enumerate(tensors):
        # Ensure the tensor is in the correct format
        if tensor.shape[-1] == 3:
            # Rearrange dimensions from (128, 128, 3) to (3, 128, 128)
            tensor = tensor.permute(2, 0, 1)
        
        # Ensure tensor is in the range [0, 1] or [0, 255]
        tensor = normalize_tensor(tensor)
        
        # Save the tensor as a PNG file
        save_image(tensor, os.path.join(path, f'image_{i}.png'))


def ensure_empty_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)