from useful_tools import generate_uniform_annular_points_in_sphere, generate_uniform_points_on_sphere, \
                         generate_uniform_annular_points_in_sphere_time_dependent,\
                         generate_uniform_annular_points_on_sphere_time_dependent

import numpy
import torch
from torch import Tensor
from problem_5_1 import f, g, Du_ft_fast, Bu_ft, h0, h1

torch.manual_seed(2) 
numpy.random.seed(2)

def MH_1_iteration_in_domain(d, R, N_inside_train, model, power_for_mh_sampling, time_dependent_type, time_interval, burn_in_propto):
    if time_dependent_type == 'none':
        x1_candidates = numpy.zeros((0, d))
    else:
        x1_candidates = numpy.zeros((0, d+1))
    RE_x1_candidates = numpy.zeros((0,))
    temp = 2 + burn_in_propto
    for e in range(temp):
        if time_dependent_type == 'none':
            x1_MH = generate_uniform_annular_points_in_sphere(d,R,10,int(N_inside_train/20))
        else:
            x1_MH = generate_uniform_annular_points_in_sphere_time_dependent(d, R, time_interval, 10, round(N_inside_train / 20))
        numpy.random.shuffle(x1_MH)
        x1_candidates = numpy.concatenate((x1_candidates, x1_MH))
        tensor_x1_MH= Tensor(x1_MH)
        tensor_x1_MH.requires_grad = False
        tensor_f1_MH = Tensor(f(x1_MH))
        tensor_f1_MH.requires_grad = False
        tensor_RE1_MH = torch.abs((Du_ft_fast(model, tensor_x1_MH) - tensor_f1_MH)) **power_for_mh_sampling
        np_RE1_MH = tensor_RE1_MH.cpu().detach().numpy()
        RE_x1_candidates = numpy.concatenate((RE_x1_candidates, np_RE1_MH))
    del tensor_x1_MH, tensor_f1_MH, tensor_RE1_MH
    len1 = len(x1_candidates)
    temp_prob_array = numpy.random.random(len1)
    range_temp = len1 -1
    for h in range(range_temp):
        a = RE_x1_candidates[h+1]
        b = RE_x1_candidates[h]
        if a < b:
            accept_prob = a/b
            if temp_prob_array[h+1] > accept_prob:
                x1_candidates[h+1] = x1_candidates[h]
    x1_candidates = x1_candidates[-N_inside_train:]
    return x1_candidates


def importance_in_domain(d, R, N_inside_train, model, power, time_dependent_type, time_interval):
    if time_dependent_type == 'none':
        x1_train = generate_uniform_annular_points_in_sphere(d, R, 10, round(N_inside_train/10))
    else:
        x1_train = generate_uniform_annular_points_in_sphere_time_dependent(d, R, time_interval, 10, round(N_inside_train / 10))
    tensor_x1_train = Tensor(x1_train)
    tensor_x1_train.requires_grad = False
    tensor_f1_train = Tensor(f(x1_train))
    tensor_f1_train.requires_grad = False
    SE_1 = torch.abs((Du_ft_fast(model, tensor_x1_train) - tensor_f1_train)) ** power # SE means residual error, not squared error.
    SSE_1 = torch.sum(SE_1)
    SE_1_np = SE_1.cpu().detach().numpy()
    SSE_1_np = SSE_1.cpu().detach().numpy()
    del SE_1, SSE_1
    prob1_np = SE_1_np / SSE_1_np
    # probability array for n uniform points
    # choose indices from old array of points for new array of output points
    inds1 = numpy.random.choice(len(x1_train), len(x1_train), p=prob1_np)
    points1 = numpy.zeros(x1_train.shape)
    for i in range(len(x1_train)):
        points1[i] = x1_train[inds1[i]]
    x1_train = points1
    return x1_train

def importance_on_IBC_time_dependent(d, R, N_boundary_train, N_initial_train, model, power, time_interval):
    x2_train, x3_train = generate_uniform_annular_points_on_sphere_time_dependent(d, R, time_interval, N_boundary_train, 10,
                                                                                  round(N_initial_train / 10))
    tensor_x2_train = Tensor(x2_train)
    tensor_x2_train.requires_grad = False
    tensor_g2_train = Tensor(g(x2_train))
    tensor_g2_train.requires_grad = False
    SE_2 = torch.abs((Bu_ft(model, tensor_x2_train) - tensor_g2_train)) ** power
    SSE_2 = torch.sum(SE_2)
    SE_2_np = SE_2.cpu().detach().numpy()
    SSE_2_np = SSE_2.cpu().detach().numpy()
    del SE_2, SSE_2
    prob2_np = SE_2_np / SSE_2_np
    # probability array for n uniform points
    # choose indices from old array of points for new array of output points
    inds2 = numpy.random.choice(len(x2_train), len(x2_train), p=prob2_np)
    points2 = numpy.zeros(x2_train.shape)
    for i in range(len(x2_train)):
        points2[i] = x2_train[inds2[i]]
    x2_train = points2

    tensor_x3_train = Tensor(x3_train)
    tensor_x3_train.requires_grad = False
    tensor_h03_train = Tensor(h0(x3_train))
    tensor_h03_train.requires_grad = False
    SE_3 = torch.abs((model(tensor_x3_train) - tensor_h03_train)) ** power # SE means residual error, not squared error.
    SSE_3 = torch.sum(SE_3)
    SE_3_np = SE_3.cpu().detach().numpy()
    SSE_3_np = SSE_3.cpu().detach().numpy()
    del SE_3, SSE_3
    prob3_np = SE_3_np / SSE_3_np
    # probability array for n uniform points
    # choose indices from old array of points for new array of output points
    inds3 = numpy.random.choice(len(x3_train), len(x3_train), p=prob3_np)
    points3 = numpy.zeros(x3_train.shape)
    for i in range(len(x3_train)):
        points3[i] = x3_train[inds3[i]]
    x3_train = points3
    return x2_train, x3_train
    
    
def importance_on_IBC(d, R, N_boundary_train, model, power):
    x2_train = generate_uniform_points_on_sphere(d,R,N_boundary_train)
    tensor_x2_train = Tensor(x2_train)
    tensor_x2_train.requires_grad = False
    tensor_g2_train = Tensor(g(x2_train))
    tensor_g2_train.requires_grad = False
    SE_2 = torch.abs((Bu_ft(model, tensor_x2_train) - tensor_g2_train)) ** power
    SSE_2 = torch.sum(SE_2)
    SE_2_np = SE_2.cpu().detach().numpy()
    SSE_2_np = SSE_2.cpu().detach().numpy()
    del SE_2, SSE_2
    prob2_np = SE_2_np / SSE_2_np
    # probability array for n uniform points
    # choose indices from old array of points for new array of output points
    inds2 = numpy.random.choice(len(x2_train), len(x2_train), p=prob2_np)
    points2 = numpy.zeros(x2_train.shape)
    for i in range(len(x2_train)):
        points2[i] = x2_train[inds2[i]]
    x2_train = points2

    return x2_train

