from utilities import true_solution, generate_points_in_the_domain, generate_points_on_the_boundary, generate_points_in_the_domain
import numpy
import torch
from torch import Tensor


def importance_on_boundary(N_2, model, p):
    x_2= generate_points_on_the_boundary(N_2)
    RE = abs(model(x_2) - true_solution(x_2)) ** p
    SRE = torch.sum(RE)
    x_2 = x_2.cpu().detach().numpy()
    RE_np = RE.cpu().detach().numpy()
    SRE_np = SRE.cpu().detach().numpy()
    prob2_np = RE_np / SRE_np
    prob2_np = prob2_np.reshape((N_2,))
    # probability array for n uniform points
    # choose indices from old array of points for new array of output points
    inds2 = numpy.random.choice(N_2, N_2, p=prob2_np)
    points2 = numpy.zeros(x_2.shape)
    for i in range(N_2):
        points2[i] = x_2[inds2[i]]
    x_2 = points2

    return Tensor(x_2)


def importance_in_domain(N_1, model, p):
    x_2= generate_points_in_the_domain(N_1, 2)
    RE = abs(model(x_2) - true_solution(x_2)) ** p
    SRE = torch.sum(RE)
    x_2 = x_2.cpu().detach().numpy()
    RE_np = RE.cpu().detach().numpy()
    SRE_np = SRE.cpu().detach().numpy()
    prob2_np = RE_np / SRE_np
    prob2_np = prob2_np.reshape((N_1,))
    # probability array for n uniform points
    # choose indices from old array of points for new array of output points
    inds2 = numpy.random.choice(N_1, N_1, p=prob2_np)
    points2 = numpy.zeros(x_2.shape)
    for i in range(N_1):
        points2[i] = x_2[inds2[i]]
    x_2 = points2

    return Tensor(x_2)





