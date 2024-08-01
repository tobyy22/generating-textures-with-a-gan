import torch

def wasserstein_loss(y_pred, real_flag):
  if real_flag:
    return -torch.mean(y_pred)
  else:
    return torch.mean(y_pred)

def similarity_loss(tensors):
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