# the solution is a solution to
# - grad(a(x) grad u) + |grad u|^2 = f, in a domain where a(x) = 1+1/2*|x|^2

import torch
import numpy
from numpy import sin, cos, zeros, pi, sqrt, absolute, ones

torch.set_default_tensor_type('torch.cuda.FloatTensor')

h = 0.01 # step length ot compute derivative

# the dimension in space
def spatial_dimension():
    return 20

# define the true solution for numpy array (N sampling points of d variables)
def true_solution(x_batch):
    r = sqrt(numpy.sum(x_batch**2,1))
    temp = absolute(1-r)
    u = sin(pi/2*temp**(2.5))    
    return u

# if the true solution is provided
def if_true_solution_given():
    return True

# the point-wise Du: Input x is a sampling point of d variables ; Output is a numpy vector which means the result of Du(x))
def Du(model,x_batch):
    s = zeros(x_batch.shape[0])
    for i in range(x_batch.shape[1]):  
        ei = zeros(x_batch.shape)
        ei[:,i] = 1
        s = s - 1/h*((1+0.5*numpy.sum((x_batch+0.5*h*ei)**2, 1))*((model.predict(x_batch+h*ei)-model.predict(x_batch))/h)\
                      - (1+0.5*numpy.sum((x_batch-0.5*h*ei)**2, 1))*((model.predict(x_batch)-model.predict(x_batch-h*ei))/h))
        s = s + ((model.predict(x_batch+h*ei)-model.predict(x_batch-h*ei))/2/h)**2
    return s

# the point-wise Du: Input x is a batch of sampling points of d variables (tensor) ; Output is tensor vector which means the result of Du(x))
def Du_ft(model,tensor_x_batch):
    s = torch.zeros(tensor_x_batch.shape[0])
    s.requires_grad=False   
    for i in range(tensor_x_batch.shape[1]):  
        ei = torch.zeros(tensor_x_batch.shape)
        ei.requires_grad=False   
        ei[:,i] = 1
        s = s - 1/h*((1+0.5*torch.sum((tensor_x_batch+0.5*h*ei)**2, 1))*((model(tensor_x_batch+h*ei)-model(tensor_x_batch))/h)\
                      - (1+0.5*torch.sum((tensor_x_batch-0.5*h*ei)**2, 1))*((model(tensor_x_batch)-model(tensor_x_batch-h*ei))/h))
        s = s + ((model(tensor_x_batch+h*ei)-model(tensor_x_batch-h*ei))/2/h)**2
    return s

def Du_ft_fast(model,tensor_x_batch):
    s = torch.zeros(tensor_x_batch.shape[0])
    s.requires_grad=False
    for i in range(tensor_x_batch.shape[1]):
        ei = torch.zeros(tensor_x_batch.shape)
        ei.requires_grad=False
        ei[:,i] = 1
        u_x_plus_hei = model(tensor_x_batch + h * ei)
        u_x = model(tensor_x_batch)
        u_x_minus_hei = model(tensor_x_batch - h * ei)
        s = s - 1/h*((1+0.5*torch.sum((tensor_x_batch+0.5*h*ei)**2, 1))*((u_x_plus_hei-u_x)/h)\
                      - (1+0.5*torch.sum((tensor_x_batch-0.5*h*ei)**2, 1))*((u_x-u_x_minus_hei)/h))
        s = s + ((u_x_plus_hei-u_x_minus_hei)/2/h)**2
    del u_x_minus_hei, u_x, u_x_plus_hei, ei
    return s


# define the right hand function for numpy array (N sampling points of d variables)
def f(x_batch):
    r = sqrt(numpy.sum(x_batch**2,1))
    temp = absolute(1-r)
    inner_part = pi/2*temp**(2.5)
    Laplace_u = -5*pi*(x_batch.shape[1]-1)/4/r*cos(inner_part)*temp**(1.5)-25*pi*pi/16*sin(inner_part)*(1-r)**3+15*pi/8*cos(inner_part)*temp**0.5
    f = 5*pi*r/4*cos(inner_part)*temp**(1.5)-(1+0.5*r**2)*Laplace_u + 25/16*pi*pi*cos(inner_part)**2*(1-r)**3
    return f

# the point-wise Bu for tensor (N sampling points of d variables)
def Bu_ft(model,tensor_x_batch):
    return model(tensor_x_batch)

# define the boundary value g for tensor (N sampling points of d variables)
def g(x_batch):
    return zeros((x_batch.shape[0],))

# the point-wise h0 for numpy array (N sampling points of d variables)
def h0(x_batch):
    return None

# the point-wise h1 for numpy array (N sampling points of d variables)
def h1(x_batch):
    return None

# specify the domain type
def domain_shape():
    return 'sphere'

# output the domain parameters
def domain_parameter(d):
    R = 1
    return R

# If this is a time-dependent problem
def time_dependent_type():
    return 'none'

# output the time interval
def time_interval():
    return None

def FD_step():
    return h