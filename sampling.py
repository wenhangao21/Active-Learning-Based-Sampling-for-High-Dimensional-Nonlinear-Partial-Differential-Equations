from utilities import true_solution, generate_points_in_the_domain, generate_points_on_the_boundary, generate_points_in_the_domain
import numpy
import torch
from torch import Tensor

def importance_on_boundary(N_2, model, p):
    device = next(model.parameters()).device  # Get device from the model
    x_2 = generate_points_on_the_boundary(N_2).to(device)  # Move points to the model's device
    RE = torch.abs(model(x_2) - true_solution(x_2)) ** p
    SRE = torch.sum(RE)
    prob2 = RE / SRE  # Compute probability on the same device
    prob2 = prob2.reshape((N_2,))

    # Perform sampling with replacement based on probabilities
    inds2 = torch.multinomial(prob2, N_2, replacement=True)
    points2 = x_2[inds2]  # Select points based on sampled indices

    return points2  # Return tensor directly on the model's device


def importance_in_domain(N_1, model, p):
    device = next(model.parameters()).device  # Get device from the model
    x_2 = generate_points_in_the_domain(N_1, 2).to(device)  # Move points to the model's device
    RE = torch.abs(model(x_2) - true_solution(x_2)) ** p
    SRE = torch.sum(RE)
    prob2 = RE / SRE  # Compute probability on the same device
    prob2 = prob2.reshape((N_1,))

    # Perform sampling with replacement based on probabilities
    inds2 = torch.multinomial(prob2, N_1, replacement=True)
    points2 = x_2[inds2]  # Select points based on sampled indices

    return points2  # Return tensor directly on the model's device





