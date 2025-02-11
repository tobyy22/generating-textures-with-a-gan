import os
import shutil
import json
import torch


from torchvision.utils import save_image

    
def normalize_tensor(tensor, normalize=True):
    if normalize:
        tensor = tensor.clone()
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor

def save_tensors_to_png(tensors, path, start_index=0):
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
        save_image(tensor, os.path.join(path, f'image_{i+start_index}.png'))


def ensure_empty_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def add_dynamic_noise_to_tensors(tensor_list, noise_ratio=0.05):
    """
    Adds dynamic noise to each tensor in the provided list.
    The noise level is based on a fraction of the tensor's standard deviation.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to which noise will be added.
        noise_ratio (float): The ratio of the tensor's standard deviation used to scale the noise.

    Returns:
        list of torch.Tensor: List of tensors with added noise.
    """
    noisy_tensors = []
    for tensor in tensor_list:
        # Compute noise level based on the tensor's standard deviation
        std_dev = tensor.std().item() if tensor.numel() > 1 else 1.0  # Avoid zero std for single-element tensors
        noise_level = std_dev * noise_ratio
        
        # Generate and add noise
        noise = torch.randn_like(tensor) * noise_level
        noisy_tensor = tensor + noise
        noisy_tensors.append(noisy_tensor)
    return noisy_tensors