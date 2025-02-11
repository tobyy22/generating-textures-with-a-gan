import torch
import torch.nn as nn

def wasserstein_loss(y_pred, real_flag, device=None):
  if real_flag:
    return -torch.mean(y_pred)
  else:
    return torch.mean(y_pred)


def similarity_loss_mse(output_original, output_perturbed):
    """
    Computes the similarity loss to penalize identical outputs and encourage diversity.

    The loss is calculated as the inverse of the Mean Squared Error (MSE) between the original
    and perturbed outputs, ensuring that identical outputs are discouraged.

    Args:
        output_original (Tensor): The original output from the U-Net generator.
        output_perturbed (Tensor): The perturbed output from the U-Net generator.

    Returns:
        Tensor: The computed similarity loss value.
    """
    mse_loss = nn.MSELoss()(output_original, output_perturbed)
    inverse_mse_loss = 1.0 / (mse_loss + 1e-6)
    return inverse_mse_loss



def bce_loss(y_pred, real_flag, device):
    criterion = torch.nn.BCELoss()

    b_size = y_pred.size(0)
    
    if real_flag:
        targets = torch.full((b_size,), 1., dtype=torch.float)
    else:
        targets = torch.full((b_size,), 0., dtype=torch.float)
    
    targets = targets.to(device)
    
    loss = criterion(y_pred, targets)
    return loss