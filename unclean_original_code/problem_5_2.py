# the solution is a solution to
# - grad(a(x) grad u) + |grad u|^2 = f, in a domain where a(x) = 1+1/2*|x|^2

import torch
import numpy
from numpy import zeros, pi, sqrt, absolute, exp, array

torch.set_default_tensor_type('torch.cuda.DoubleTensor')

h = 0.01 # step length ot compute derivative

# the dimension in space
def spatial_dimension():
    return 20

def true_solution(x_batch):
    u = exp(sqrt(1-x_batch[:,0])*sqrt(numpy.sum(x_batch[:,1:]**2,1)))
    return u

# if the true solution is provided
def if_true_solution_given():
    return True

# Du is on cpu
# the point-wise Du: Input x is a sampling point of d variables ; Output is a numpy vector which means the result of Du(x))
def Du(model,x_batch):
    s = zeros(x_batch.shape[0],)
    for i in range(x_batch.shape[1]):  
        ei = zeros(x_batch.shape)
        ei[:,i] = 1
        if i == 0:
            indices = x_batch[:,0]<1-h-1e-10
            s[indices] = s[indices] + (model.predict(x_batch[indices,:]+h*ei[indices,:])-model.predict(x_batch[indices,:]-h*ei[indices,:]))/2/h
            indices = x_batch[:,0]>=1-h-1e-10
            s[indices] = s[indices] + (3*model.predict(x_batch[indices,:])-4*model.predict(x_batch[indices,:]-h*ei[indices,:])+model.predict(x_batch[indices,:]-2*h*ei[indices,:]))/2/h
        else:
            s = s - 1/h*((1+0.5*numpy.sum((x_batch[:,1:]+0.5*h*ei[:,1:])**2, 1))*((model.predict(x_batch+h*ei)-model.predict(x_batch))/h)\
                 - (1+0.5*numpy.sum((x_batch[:,1:]-0.5*h*ei[:,1:])**2, 1))*((model.predict(x_batch)-model.predict(x_batch-h*ei))/h))
    return s

# Du_ft is on gpu
# the point-wise Du: Input x is a batch of sampling points of d variables (tensor) ; Output is tensor vector which means the result of Du(x))
def Du_ft(model,tensor_x_batch):
    s = torch.zeros(tensor_x_batch.shape[0],)
    for i in range(tensor_x_batch.shape[1]):  
        ei = torch.zeros(tensor_x_batch.shape)
        ei[:,i] = 1
        if i == 0:
            indices = tensor_x_batch[:,0]<1-h-1e-10
            s[indices] = s[indices] + (model(tensor_x_batch[indices,:]+h*ei[indices,:])-model(tensor_x_batch[indices,:]-h*ei[indices,:]))/2/h
            indices = tensor_x_batch[:,0]>=1-h-1e-10
            s[indices] = s[indices] + (3*model(tensor_x_batch[indices,:])-4*model(tensor_x_batch[indices,:]-h*ei[indices,:])+model(tensor_x_batch[indices,:]-2*h*ei[indices,:]))/2/h
        else:
            s = s - 1/h*((1+0.5*torch.sum((tensor_x_batch[:,1:]+0.5*h*ei[:,1:])**2, 1))*((model(tensor_x_batch+h*ei)-model(tensor_x_batch))/h)\
                  - (1+0.5*torch.sum((tensor_x_batch[:,1:]-0.5*h*ei[:,1:])**2, 1))*((model(tensor_x_batch)-model(tensor_x_batch-h*ei))/h))
    return s


def Du_ft_fast(model,tensor_x_batch):
    s = torch.zeros(tensor_x_batch.shape[0],)
    for i in range(tensor_x_batch.shape[1]):
        ei = torch.zeros(tensor_x_batch.shape)
        ei[:,i] = 1
        if i == 0:
            indices = tensor_x_batch[:,0]<1-h-1e-10
            s[indices] = s[indices] + (model(tensor_x_batch[indices,:]+h*ei[indices,:])-model(tensor_x_batch[indices,:]-h*ei[indices,:]))/2/h
            indices = tensor_x_batch[:,0]>=1-h-1e-10
            s[indices] = s[indices] + (3*model(tensor_x_batch[indices,:])-4*model(tensor_x_batch[indices,:]-h*ei[indices,:])+model(tensor_x_batch[indices,:]-2*h*ei[indices,:]))/2/h
        else:
            modelx = model(tensor_x_batch)
            s = s - 1/h*((1+0.5*torch.sum((tensor_x_batch[:,1:]+0.5*h*ei[:,1:])**2, 1))*((model(tensor_x_batch+h*ei)-modelx)/h)\
                  - (1+0.5*torch.sum((tensor_x_batch[:,1:]-0.5*h*ei[:,1:])**2, 1))*((modelx-model(tensor_x_batch-h*ei))/h))
    return s

# define the right hand function for numpy array (N sampling points of d variables)
def f(x_batch):
    d = x_batch.shape[1]-1
    t = x_batch[:,0]
    r = sqrt(numpy.sum(x_batch[:,1:]**2,1))
    exp_term = exp(sqrt(1-t)*r)
    f = -r/2/sqrt(1-t)*exp_term - exp_term*sqrt(1-t)*r - (1+0.5*r**2)*exp_term*(1-t+sqrt(1-t)/r*(d-1))

    return f

# the point-wise Bu for tensor (N sampling points of d variables)
def Bu_ft(model,tensor_x_batch):
    return model(tensor_x_batch)

# define the boundary value g for tensor (N sampling points of d variables)
def g(x_batch):
    g = true_solution(x_batch)
    return g

# the point-wise h0 for numpy array (N sampling points of d variables)
def h0(x_batch):
    h0 = exp(sqrt(numpy.sum(x_batch[:,1:]**2,1)))
    return h0

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
    return 'parabolic'

# output the time interval
def time_interval():
    return array([0,1])

def FD_step():
    return h