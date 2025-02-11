import torch

def wasserstein_loss(y_pred, real_flag, device=None):
  if real_flag:
    return -torch.mean(y_pred)
  else:
    return torch.mean(y_pred)


def similarity_loss_old(tensors):
    # Initialize a list to store the distances
    distances = []

    # Iterate over the tensors
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            # Compute the distance between the tensors
            distance = torch.sqrt(torch.sum((tensors[i] - tensors[j]) ** 2))

            # Add the distance to the list
            distances.append(distance)

    mean_distance = torch.mean(torch.tensor(distances))

    # Return the list of distances
    return mean_distance.requires_grad_()

def similarity_loss(tensors, koef=1.):
    # Initialize a list to store the distances
    distances = []

    # Iterate over the tensors
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            # Compute the distance between the tensors
            distance = torch.sum((tensors[i] - tensors[j]) ** 2)

            # Add the distance to the list
            distances.append(distance)

    mean_distance = torch.mean(torch.tensor(distances))

    # similary

    # Return the list of distances
    return (koef*(1 / (mean_distance + 1e-8))).requires_grad_()



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