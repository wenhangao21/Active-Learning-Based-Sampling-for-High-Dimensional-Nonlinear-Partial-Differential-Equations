import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
import timeit
import numpy
from numpy import zeros, sum, sqrt, linspace, absolute
from utilities import generate_points_in_the_domain, generate_points_on_the_boundary, gradients, weights_init, true_solution,\
                      get_l2_error, l, f, get_testing_points
import network_setting
import scipy.io
from sampling import importance_on_boundary, importance_in_domain

# set to be run on cuda
TensorType = 'Double'
if TensorType == 'Double':
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
elif TensorType == 'Float':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# 1 some hyper-parameters
n_epoch = 50000 # number of epochs  approximately 0.024 sec per epoch
trials = 30  # number of times run the program with different random seeds to get results
end_time = 300  # how many seconds wanna run

N_1 = 1024  # number of training points in the interior
N_2 = 1024  # number of training points on the boundary, better divisible by 2^2 = 4
m = 100  # number of nodes in each layer for the basic and error sampling
d = 2  # dimension of the problem, which should be 2, do not change
m_drm_dgm = 20  # number of nodes in each layer for drm, dgm
lr = 3e-3  # learning rates
lambda_term = 5000000  # boundary weighting term
l2error_different_seeds = zeros((trials,))   # create a sequence to store final l2error
time_different_seeds = zeros((trials,))  # create a sequence to store final time, doesn't really matter tho
x_test = get_testing_points()  # testing points, grid points in the square domain, step size 0.01
true_solution_test = true_solution(x_test)  # true solution at x_test

def er():
    error_dict = []  # dictionary to store different error/time sequences
    time_dict = []
    for k in range(trials):
        print('trial: ', k, '\n')
        torch.manual_seed(k+1)
        numpy.random.seed(k+1)
        model = network_setting.drnn(m_drm_dgm)
        # model.apply(weights_init)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.99)
        time = 0
        timeseq = zeros((n_epoch,))
        l2seq = zeros((n_epoch,))
        for epoch in range(n_epoch):
            if time > end_time:     # break at end_time seconds and store the final error
                l2error_different_seeds[k] = l2_error
                time_different_seeds[k] = time
                break
            tic = timeit.default_timer()
            # 1 for interior and 2 for boundary
            x_1 = generate_points_in_the_domain(N_1, d)
            x_2 = importance_on_boundary(N_2, model, 1)
            x_1.requires_grad_()
            network_solution_1 = model(x_1)
            network_solution_2 = model(x_2)
            true_solution_2 = true_solution(x_2)
            nabla = autograd.grad(outputs=network_solution_1, inputs=x_1,
                                  grad_outputs=torch.ones_like(network_solution_1),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
            delta = gradients(x_1, nabla[:, 0: 1])[:, 0: 1] + gradients(x_1, nabla[:, 1: 2])[:, 1: 2]
            loss_1 = torch.mean(torch.square(delta - 2)) # mean squared error, -delta - (-2) = -(delta - 2)
            loss_2 = lambda_term * torch.mean(torch.square(network_solution_2 - true_solution_2)) # mean absolute error
            loss = loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            StepLR.step()
            toc = timeit.default_timer()
            time_one_epoch = toc - tic
            time = time + time_one_epoch
            timeseq[epoch] = time
            l2_error = get_l2_error(model(x_test), true_solution_test)
            l2_error = l2_error.cpu().detach().numpy()
            l2seq[epoch] = l2_error
            if epoch % 500 == 0:
                print('time: ', time,  'epoch:', epoch, 'loss:', loss.item(), 'err:', l2_error, '\n')
        error_dict.append(l2seq)
        time_dict.append(timeseq)
        scipy.io.savemat('l2seq'+ str(k) + '.mat', mdict={str(k) + 'l2seq': l2seq})
        scipy.io.savemat('timeseq'+ str(k) + '.mat', mdict={str(k) + 'timeseq': timeseq})
    l2_average_seq = zeros((n_epoch,))
    time_average_seq = zeros((n_epoch,))
    for k in range(trials):
        l2_average_seq = l2_average_seq  + error_dict[k]
        time_average_seq = time_average_seq + time_dict[k]
    l2_average_seq = l2_average_seq/trials
    time_average_seq = time_average_seq/trials
    #  store data in .mat matlab file
    scipy.io.savemat('er_average_error.mat', mdict={'er_average_error': l2_average_seq})
    scipy.io.savemat('er_average_time.mat', mdict={'er_average_time': time_average_seq})
    scipy.io.savemat('er_l2error_different_seeds.mat', mdict={'er_l2error_different_seeds': l2error_different_seeds})
    scipy.io.savemat('er_time_different_seeds.mat', mdict={'er_time_different_seeds': time_different_seeds})


if __name__ == '__main__':
    er()




