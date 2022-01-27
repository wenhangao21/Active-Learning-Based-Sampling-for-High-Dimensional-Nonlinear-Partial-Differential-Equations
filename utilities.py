import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd, Tensor
import numpy

TensorType = 'Double'
if TensorType == 'Double':
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
elif TensorType == 'Float':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


# return a tensor of random points in the interior of the domain
def generate_points_in_the_domain(N_1, d):
    return torch.rand(N_1, d)


# return a tensor of random points on the bdy
def generate_points_on_the_boundary(N_2):
    num = -(-N_2//4)  # get ceil of N_2/4,
    zero_to_one = torch.rand(num, 1)
    # x1-4 for 4 sides of the rectangle, each num points
    x1 = torch.cat((zero_to_one, torch.zeros_like(zero_to_one)), dim=1)
    x2 = torch.cat((zero_to_one, torch.ones_like(zero_to_one)), dim=1)
    x3 = torch.cat((torch.ones_like(zero_to_one), zero_to_one ), dim=1)
    x4 = torch.cat((torch.zeros_like(zero_to_one), zero_to_one ), dim=1)
    x = torch.cat((x1, x2, x3, x4), dim=0)
    return x


# calculate nabla u
def gradients(input, output):
    return autograd.grad(outputs=output, inputs=input,
                                grad_outputs=torch.ones_like(output),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

# initialization in the paper, will not be used for comparison purpose
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


# true solution of u(x)
def true_solution(x):
    # if x > 0.5, u(x) = (1-x)^2. u = x^2 otherwise
    value = torch.where(x[:, 0: 1] > 0.5, (1 - x[:, 0: 1]) ** 2, x[:, 0: 1] ** 2)
    return value


# return a tensor of shape 1, 1, overall l2 error
def get_l2_error(network_solution, true_solution):
    return torch.norm(network_solution - true_solution) / torch.norm(true_solution)


# only for wan
def l(x):
    # for wan
    return x[:, 0: 1] * x[:, 1: 2] * (1 - x[:, 0: 1]) * (1 - x[:, 1: 2])


# function f
def f():
    # f(x) = -2
    return -2

# generate testing grid points for l2 error, return tensor of 9801 grid points
def get_testing_points():
    y_coor = numpy.arange(0.01, 1, 0.01)
    x_test_numpy = numpy.array([[x0, y0] for x0 in y_coor for y0 in y_coor])
    x_test = Tensor(x_test_numpy)
    x_test.requires_grad = False
    return x_test





