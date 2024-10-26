# the main codes for the residual method to solve PDEs

import torch
from torch import Tensor, optim
import numpy
import scipy.io
from numpy import zeros, sum, sqrt, linspace, absolute
from useful_tools import generate_uniform_points_in_cube, generate_uniform_points_in_cube_time_dependent,\
    generate_uniform_points_on_cube, generate_uniform_points_on_cube_time_dependent,\
    generate_uniform_points_in_sphere, generate_uniform_points_in_sphere_time_dependent,\
    generate_uniform_points_on_sphere, generate_uniform_points_on_sphere_time_dependent,\
    generate_uniform_annular_points_in_sphere_time_dependent, generate_uniform_annular_points_on_sphere_time_dependent,\
    generate_uniform_annular_points_in_sphere, generate_learning_rates
import SelectNet_setting
import pickle
import time
import timeit

# rembem to change problem number in active_sampling as well
from problem_5_1 import spatial_dimension, true_solution, if_true_solution_given, Du, Du_ft_fast, Bu_ft, f, g, h0, h1, domain_shape, \
    domain_parameter, time_dependent_type, time_interval, FD_step
from active_sampling import MH_1_iteration_in_domain, importance_in_domain, importance_on_IBC_time_dependent, importance_on_IBC
# rembem to change problem number in active_sampling as well

answer = input('Did you change path and problem number in active_sampling.py? Answer yes or no')
if answer.lower().startswith("n"):
      print("change problem number and re-run the program")
      exit()
# Run on cuda(GPU), if your GPU does not support CUDA, try google colab
TensorType = 'Double'
torch.manual_seed(1) 
numpy.random.seed(1)
if TensorType == 'Double':
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
elif TensorType == 'Float':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

########### Set parameters #############
# 0
method = 'B' # choose methods: B(basic), S(SelectNet)
sampling = 'none'  # 'self-normalized' or 'MH_sampling' or 'none'
power_for_ip_sampling = 10
power_for_mh_sampling = 8
burn_in_propto = 2  # number of burns = burn_in_propto/2 * batch_size, e.g., burn_in_prop = 2 means #burn-ins=batch_size
sampling_on_IBC = 'self-normalized'  # 'self-normalized' or 'MH_sampling' or 'none'     # IBC means initial/boundary conditions
power_for_ip_sampling_on_IBC = 20
power_for_mh_sampling_on_IBC = 4
burn_in_propto_on_IBC = 2


#1
d = spatial_dimension()  # dimension of problem

L = 3 # depth of the solution network
m = 100  # number of nodes in each layer of solution network

#2
n_epoch = 20000  # number of outer iterations
N_inside_train = 12000 # number of trainning sampling points inside the domain in each epoch (batch size)
# please set N_inside_train \geq 16 for numerical stability. this program is written with the assumption that N_inside_train  is a power of 2
lrseq = generate_learning_rates(-3,-6,n_epoch,0.999,1000)
lambda_term = 10

#3 for saving files, empty if save in the same directory,  to train on google colab put in 'drive/MyDrive/6_1_2/' ,where 6_1_2 is my folder name
path = ''

if method == 'S' or method == 'RS':
    m_sn = 20   #number of nodes in each layer of the selection network
    penalty_parameter = 0.001
    maxvalue_sn = 5  # upper bound for the selection net weighting
    minvalue_sn = 0.8  # lower bound for the selection net
    lr_sn = 0.0001  # learning rate for the selection net
    # n_update_each_batch_sn = 1
    loss_sn1_threshold = 1e-8  # stopping criteria for training the selection net1 (inside the domain)
    loss_sn2_threshold = 1e-8  # stopping criteria for training the selection net2 (boudanry or initial)
    selectnet_initial_constant = 1.0  # if selectnet is initialized as constant one
activation = 'ReLU3'  # activation function for the solution net
boundary_control = 'none'  # 'homo_unit_sphere'   if the solution net architecture satisfies the boundary condition automatically
initial_control = 'none'  # 'homo_parabolic' for 5_2, 'homo_hyperbolic' for 5_3, if the solution net architecture satisfies the initial condition automatically
initial_constant = 'none' # initial value for the solution network

## sampling type will always be uniform annular, since it's a better biasing distribution than uniform
# sphere_sampling_type = 'uniform_annular' # 'uniform', 'uniform_annular'
# test_type = 'uniform_annular' # 'uniform', 'uniform_annular'

########### Set problem parameters   #############
h_Du_t = FD_step()  # time length for computing the first derivative of t by finite difference (for the hyperbolic equations)
FD_step = FD_step()
time_dependent_type = time_dependent_type()   ## If this is a time-dependent problem
domain_shape = domain_shape()  ## the shape of domain
## domain_shape == 'sphere':
R = domain_parameter(d)
domain_intervals = []
if not time_dependent_type == 'none':    
    time_interval = time_interval()
else:
    time_interval = []

# print and save setting to text file for record.
str_details = 'sampling: ' + sampling +' sampling_on_IBC: ' + sampling_on_IBC + ' method: ' + str(method) \
              + '\n power_ip: ' + str(power_for_ip_sampling) + ' power_mh: ' + str(power_for_mh_sampling) \
              +'\n power_ip_on_IBC: ' + str(power_for_ip_sampling_on_IBC) + ' power_mh_on_IBC: ' + str(power_for_mh_sampling_on_IBC) \
              + '\n dimension: ' + str(d) + ' epoch: ' + str(n_epoch) + ' N_inside: ' + str(N_inside_train)  \
              + '\n bdy_control: ' + boundary_control + ' ini_control: ' + initial_control + ' ini_constant: ' + initial_constant + '\n'

print(str_details)
log_file = open(path + "log.txt", "a")
text_file = open(path + "detail.txt", "w")
n = text_file.write(str_details)
text_file.close()
    
########### Interface parameters #############
flag_compute_loss_each_epoch = True # if compute loss after each epoch
n_epoch_show_info = max([round(n_epoch/50),1]) # how many epoch will it show information
flag_output_results = True # if save the results as files in current directory

#4
# test points: in entire \Omega;  given pts: in B_{0.1}(0)
if if_true_solution_given() == True: # if we need to compute the error
    N_test = 10000 # number of testing points
    flag_l2error = True
    flag_maxerror = True
    flag_givenpts_l2error = False
    flag_givenpts_maxerror = False
    if flag_givenpts_l2error == True or flag_givenpts_maxerror == True:
        if time_dependent_type == 'none':
            given_pts = generate_uniform_points_in_sphere(d,0.1,10000)
            givenpts_info = 'given_pts = generate_uniform_points_in_sphere(d,0.1,10000)'
        else:
            given_pts = generate_uniform_points_in_sphere_time_dependent(d,0.1,time_interval,10000)
    
    
########### Depending parameters #############
if time_dependent_type == 'none':
    u_net = SelectNet_setting.network(d, m, L, activation_type = activation, boundary_control_type = boundary_control, initial_constant = initial_constant)
else:
    u_net = SelectNet_setting.network_time_depedent(d, m, L, activation_type = activation, boundary_control_type = boundary_control, initial_constant = initial_constant)
if method == 'S' or method == 'RS':
    if time_dependent_type == 'none':
        select_net1 = SelectNet_setting.selection_network(d,m_sn,maxvalue_sn,minvalue_sn, initial_constant = selectnet_initial_constant) # selection_network inside the domain
        select_net2 = SelectNet_setting.selection_network(d,m_sn,maxvalue_sn,minvalue_sn, initial_constant = selectnet_initial_constant)  # selection_network for initial and boudanry conditions
    else:
        select_net1 = SelectNet_setting.selection_network(d+1,m_sn,maxvalue_sn,minvalue_sn, initial_constant = selectnet_initial_constant)  # selection_network for initial and boudanry conditions
        select_net2 = SelectNet_setting.selection_network(d+1,m_sn,maxvalue_sn,minvalue_sn, initial_constant = selectnet_initial_constant)  # selection_network for initial and boudanry conditions



if u_net.if_boundary_controlled == False:
    flag_boundary_term_in_loss = True  # if loss function has the boundary residual
else:
    flag_boundary_term_in_loss = False

if time_dependent_type == 'none':
    flag_initial_term_in_loss = False  # if loss function has the initial residual
else:
    if u_net.if_initial_controlled == False:
        flag_initial_term_in_loss = True
    else:
        flag_initial_term_in_loss = False

if flag_boundary_term_in_loss == True or flag_initial_term_in_loss == True:
    flag_IBC_in_loss = True  # if loss function has the boundary/initial residual
    N_IBC_train = 0  # number of boundary and initial training points
else:
    flag_IBC_in_loss = False

if flag_boundary_term_in_loss == True:
    if d == 1 and time_dependent_type == 'none':
        N_boundary_train = 2
    else:
        N_boundary_train = N_inside_train # number of sampling points on each domain face when training
    N_IBC_train = N_IBC_train + N_boundary_train
else:
    N_each_face_train = 0
    N_boundary_train = 0

if flag_initial_term_in_loss == True:          
    N_initial_train = max([1,int(round(N_inside_train/d))]) # number of sampling points on each domain face when training
    N_IBC_train = N_IBC_train + N_initial_train

# calculate partial u partial t for hyperbolic
if not time_dependent_type == 'none':   
    def Du_t_ft(model, tensor_x_batch):
        h = h_Du_t # step length ot compute derivative
        s = torch.zeros(tensor_x_batch.shape[0])
        ei = torch.zeros(tensor_x_batch.shape)
        ei[:,0] = 1
        s = (3*model(tensor_x_batch+2*h*ei)-4*model(tensor_x_batch+h*ei)+model(tensor_x_batch))/2/h
        return s
    
#################### Start Training ######################
optimizer = optim.Adam(u_net.parameters(),lr=lrseq[0])
if method == 'S' or method == 'RS':
    optimizer_sn1 = optim.Adam(select_net1.parameters(),lr=lr_sn)
    if flag_IBC_in_loss == True:
        optimizer_sn2 = optim.Adam(select_net2.parameters(),lr=lr_sn)

lossseq = zeros((n_epoch,))
resseq = zeros((n_epoch,))
l2errorseq = zeros((n_epoch,))
maxerrorseq = zeros((n_epoch,))
givenpts_l2errorseq = zeros((n_epoch,))
givenpts_maxerrorseq = zeros((n_epoch,))
time_seq = zeros((n_epoch,))

if if_true_solution_given() == True:
    if time_dependent_type == 'none':
        x_test = generate_uniform_annular_points_in_sphere(d,R,10,round(N_test/10))
    else:
        x_test = generate_uniform_annular_points_in_sphere_time_dependent(d,R,time_interval,10,round(N_test/10))

############ Precomputation ############
if if_true_solution_given() == True:
    if flag_l2error == True:
        u_x_test = true_solution(x_test)
        u_l2norm_x_test = sqrt(sum(u_x_test**2)/x_test.shape[0])
    if flag_maxerror == True:
        u_x_test = true_solution(x_test)
        u_maxnorm_x_test = numpy.max(absolute(u_x_test))
    if flag_givenpts_l2error == True:
        u_givenpts = true_solution(given_pts)
        u_l2norm_givenpts = sqrt(sum(u_givenpts**2)/given_pts.shape[0])
    if flag_givenpts_maxerror == True:
        u_givenpts = true_solution(given_pts)
        u_maxnorm_givenpts = numpy.max(absolute(u_givenpts))       
           

############ Local Time ###############    
# Training
k = 0       
time = 0
while k < n_epoch:
    ## generate training and testing data (the shape is (N,d)) or (N,d+1) 
    ## label 1 is for the points inside the domain, 2 is for those on the bondary or at the initial time   
    tic = timeit.default_timer()
    if time_dependent_type == 'none':
        if sampling == 'self-normalized':
            x1_train = importance_in_domain(d, R, N_inside_train, u_net, power_for_ip_sampling, time_dependent_type, time_interval)
        elif sampling == 'MH_sampling':
            x1_train = MH_1_iteration_in_domain(d, R, N_inside_train, u_net, power_for_mh_sampling, time_dependent_type, time_interval, burn_in_propto)
        else:
            x1_train = generate_uniform_annular_points_in_sphere(d,R,10,round(N_inside_train/10))
        if flag_IBC_in_loss == True:
            if sampling_on_IBC == 'self-normalized':
                x2_train = importance_on_IBC(d, R, N_boundary_train, u_net, power_for_ip_sampling_on_IBC)
            else:
                x2_train = generate_uniform_points_on_sphere(d,R,N_boundary_train)
    else:
        if sampling == 'self-normalized':
            x1_train = importance_in_domain(d, R, N_inside_train, u_net, power_for_ip_sampling, time_dependent_type, time_interval)
        elif sampling == 'MH_sampling':
            x1_train = MH_1_iteration_in_domain(d, R, N_inside_train, u_net, power_for_mh_sampling, time_dependent_type, time_interval, burn_in_propto)
        else:
            x1_train = generate_uniform_annular_points_in_sphere_time_dependent(d,R,time_interval,10,round(N_inside_train/10))
        if flag_IBC_in_loss == True:
            # x2_train for boudanry samplings; x3_train for initial samplings
            if flag_boundary_term_in_loss == False:
                [], x3_train = generate_uniform_annular_points_on_sphere_time_dependent(d,R,time_interval,0,10,round(N_initial_train/10))
            elif flag_initial_term_in_loss == False:
                x2_train, [] = generate_uniform_annular_points_on_sphere_time_dependent(d,R,time_interval,N_boundary_train,0,0)
            else:
                if sampling_on_IBC == 'self-normalized':
                    x2_train, x3_train = importance_on_IBC_time_dependent(d, R, N_boundary_train, N_initial_train, u_net,
                                                                          power_for_ip_sampling_on_IBC, time_interval)
                else:
                    x2_train, x3_train = generate_uniform_annular_points_on_sphere_time_dependent(d,R,time_interval,N_boundary_train,10,
                                                                                              round(N_initial_train/10))
        
    tensor_x1_train = Tensor(x1_train)
    tensor_x1_train.requires_grad=False
    tensor_f1_train = Tensor(f(x1_train))
    tensor_f1_train.requires_grad=False
    if flag_boundary_term_in_loss == True:
        tensor_x2_train = Tensor(x2_train)
        tensor_x2_train.requires_grad=False
        tensor_g2_train = Tensor(g(x2_train))
        tensor_g2_train.requires_grad=False
        
    if flag_initial_term_in_loss == True:
        tensor_x3_train = Tensor(x3_train)
        tensor_x3_train.requires_grad=False
        if time_dependent_type == 'parabolic' or time_dependent_type == 'hyperbolic':
            # h0 for u(x,0) = h0(x)
            tensor_h03_train = Tensor(h0(x3_train))
            tensor_h03_train.requires_grad=False
        if time_dependent_type == 'hyperbolic':
            # h1 for u_t(x,0) = h1(x)
            tensor_h13_train = Tensor(h1(x3_train))
            tensor_h13_train.requires_grad=False
           
    ## Set learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrseq[k]
        
    ## Train the selection net inside the domain
    if method == 'S' or method == 'RS':
        const_tensor_loss_term_x1_train = (Du_ft_fast(u_net,tensor_x1_train)-tensor_f1_train)**2
        old_loss = 1e5
        ## Compute the loss
        if method == 'S':
            loss_sn = -1/torch.sum(const_tensor_loss_term_x1_train)*torch.sum((select_net1(tensor_x1_train)*const_tensor_loss_term_x1_train)) + 1/penalty_parameter*(1/N_inside_train*torch.sum(select_net1(tensor_x1_train))-1).pow(2)
        elif method == 'RS':
            loss_sn = 1/torch.sum(const_tensor_loss_term_x1_train)*torch.sum((select_net1(tensor_x1_train)*const_tensor_loss_term_x1_train)) + 1/penalty_parameter*(1/N_inside_train*torch.sum(select_net1(tensor_x1_train))-1).pow(2)

        ## If loss_sn is stable, then break
        if abs(loss_sn.item()-old_loss)<loss_sn1_threshold:
            break
        old_loss = loss_sn.item()
        ## Update the network
        optimizer_sn1.zero_grad()
        loss_sn.backward(retain_graph=False)
        optimizer_sn1.step()
            
    ## Train the selection net on the boudnary or on the initial slice
    if method == 'S' or method == 'RS':
        if flag_IBC_in_loss == True:
            if flag_boundary_term_in_loss == True:
                const_tensor_residual_square_x2_train = (Bu_ft(u_net,tensor_x2_train)-tensor_g2_train)**2
            if flag_initial_term_in_loss == True:
                const_tensor_residual_square_x3_train = (u_net(tensor_x3_train)-tensor_h03_train)**2
            old_loss = 1e5
            ## Compute the loss
            loss_sn = Tensor([0])
            tensor_IBC_sum_term = Tensor([0])
            if flag_boundary_term_in_loss == True:
                if method == 'S':
                    loss_sn = loss_sn - 1/torch.sum(const_tensor_residual_square_x2_train)*torch.sum((select_net2(tensor_x2_train)*const_tensor_residual_square_x2_train))
                elif method == 'RS':
                    loss_sn = loss_sn + 1/torch.sum(const_tensor_residual_square_x2_train)*torch.sum((select_net2(tensor_x2_train)*const_tensor_residual_square_x2_train))
                tensor_IBC_sum_term = tensor_IBC_sum_term + torch.sum(select_net2(tensor_x2_train))
            if flag_initial_term_in_loss == True:
                if method == 'S':
                    loss_sn = loss_sn - 1/torch.sum(const_tensor_residual_square_x3_train)*torch.sum((select_net2(tensor_x3_train)*const_tensor_residual_square_x3_train))
                elif method == 'RS':
                    loss_sn = loss_sn + 1/torch.sum(const_tensor_residual_square_x3_train)*torch.sum((select_net2(tensor_x3_train)*const_tensor_residual_square_x3_train))
                tensor_IBC_sum_term = tensor_IBC_sum_term + torch.sum(select_net2(tensor_x3_train))

            loss_sn = loss_sn + 1/penalty_parameter*(1/N_IBC_train*tensor_IBC_sum_term-1).pow(2)
            ## If loss_sn is stable, then break
            if abs(loss_sn.item()-old_loss)<loss_sn2_threshold:
                break
            old_loss = loss_sn.item()
            ## Update the network
            optimizer_sn2.zero_grad()
            loss_sn.backward(retain_graph=True)
            optimizer_sn2.step()
        
    ## Train the solution net
    if method == 'S' or method == 'RS':
        const_tensor_sn_x1_train = select_net1(tensor_x1_train)
        if flag_boundary_term_in_loss == True:
            const_tensor_sn2_x2_train = select_net2(tensor_x2_train)
        if flag_initial_term_in_loss == True:
            const_tensor_sn2_x3_train = select_net2(tensor_x3_train)

    if method == 'B':
        loss1 = 1/N_inside_train*torch.sum((Du_ft_fast(u_net,tensor_x1_train)-tensor_f1_train)**2)
    elif method == 'S' or method == 'RS':
        loss1 = 1/N_inside_train*torch.sum(const_tensor_sn_x1_train*(Du_ft_fast(u_net,tensor_x1_train)-tensor_f1_train)**2)
    loss = loss1

    if flag_IBC_in_loss == True:
        loss2 = Tensor([0])
        if flag_boundary_term_in_loss == True:
            if method == 'B':
                loss2 = loss2 + torch.sum((Bu_ft(u_net,tensor_x2_train)-tensor_g2_train)**2)
            elif method == 'S' or method == 'RS':
                loss2 = loss2 + torch.sum(const_tensor_sn2_x2_train*(Bu_ft(u_net,tensor_x2_train)-tensor_g2_train)**2)
        if flag_initial_term_in_loss == True:
            if method == 'B':
                loss2 = loss2 + torch.sum((u_net(tensor_x3_train)-tensor_h03_train)**2)
            elif method == 'S' or method == 'RS':
                loss2 = loss2 + torch.sum(const_tensor_sn2_x3_train*(u_net(tensor_x3_train)-tensor_h03_train)**2)
            if time_dependent_type == 'hyperbolic':
                    loss2 = loss2 + torch.sum((Du_t_ft(u_net,tensor_x3_train)-tensor_h13_train)**2)
        loss2 = lambda_term/N_IBC_train*loss2
        loss = loss1 + loss2

    ## Update the network
    optimizer.zero_grad()
    loss.backward(retain_graph= False)
    optimizer.step()
    toc = timeit.default_timer()
    time_one_epoch = toc - tic
    time = time + time_one_epoch
    time_seq[k] = time
            
    # Save loss and L2 error
    lossseq[k] = loss.item()
    if if_true_solution_given() == True:
        if flag_l2error == True:
            l2error = sqrt(sum((u_net.predict(x_test) - u_x_test)**2)/x_test.shape[0])/u_l2norm_x_test
            l2errorseq[k] = l2error
        if flag_maxerror == True:
            maxerror = numpy.max(absolute(u_net.predict(x_test) -u_x_test))/u_maxnorm_x_test
            maxerrorseq[k] = maxerror
        if flag_givenpts_l2error == True:
            givenpts_l2error = sqrt(sum((u_net.predict(given_pts) - u_givenpts)**2)/given_pts.shape[0])/u_l2norm_givenpts
            givenpts_l2errorseq[k] = givenpts_l2error
        if flag_givenpts_maxerror == True:
            givenpts_maxerror = numpy.max(absolute(u_net.predict(given_pts) -u_givenpts))/u_maxnorm_givenpts
            givenpts_maxerrorseq[k] = givenpts_maxerror
    resseq[k] = sqrt(1/N_inside_train*sum((Du(u_net,x1_train)-f(x1_train))**2))
    
    ## Print information
    if k%n_epoch_show_info==0:
        if flag_compute_loss_each_epoch:
            print("epoch = %d, loss = %2.6f" %(k,loss.item()), end='')
            log_file.write('epoch: ' + str(k))
            print(", lr = %2.6e" %lrseq[k], end='')
        if if_true_solution_given() == True:
            if flag_l2error == True:
                print(", l2 error = %2.6e" % l2error, end='')
                log_file.write(', l2 error = ' + str(l2error))
            if flag_maxerror == True:
                print(", max error = %2.6e" % maxerror, end='')
            if flag_givenpts_l2error == True:
                print(", givenpts l2 error = %2.3e" % givenpts_l2error, end='')
                log_file.write(', givenpts l2 error = ' + str(givenpts_l2error))
            if flag_givenpts_maxerror == True:
                print(", givenpts max error = %2.3e" % givenpts_maxerror, end='')
        else:
            print("residual = %2.3e" % resseq[k], end='')
        if method == 'S':
            print(", max(select_net(x1_train)) = %2.3e" % numpy.max(select_net1.predict(x1_train)), end='')
        print("\n")
        log_file.write("\n")
    if k > 0 and k % 2000 == 0:
        # save the temp data
        torch.save(u_net.state_dict(), path + 'unet_' + str(k) + '.pt')
        scipy.io.savemat(path + 'givenpts_l2errorseq' + str(k) + '.mat', mdict={'givenpts_l2errorseq': givenpts_l2errorseq})
        scipy.io.savemat(path + 'testing_l2errorseq' + str(k) + '.mat', mdict={'l2errorseq': l2errorseq})
        if method == 'S':
            torch.save(select_net1.state_dict(), path + 'sn1para_temp_' + str(k) + '.pt')
            if flag_IBC_in_loss == True:
                torch.save(select_net2.state_dict(), path + 'sn2para_temp_' + str(k) + '.pt')
    # increment
    k = k + 1

# Save u_net
if flag_output_results == True:    
    torch.save(u_net.state_dict(), path + 'networkpara_'+'.pt')
    if method == 'S' or method == 'RS':
        torch.save(select_net1.state_dict(), path + 'select_networkpara_'+'.pt')
        torch.save(select_net2.state_dict(), path + 'select_network2para_temp_'+'.pt')

# Output results
if flag_output_results == True:    
    #save the data
    main_file_name = 'file_name'
    data = {'main_file_name':main_file_name,\
                                'givenpts_l2errorseq':givenpts_l2errorseq,\
                                'givenpts_maxerrorseq':givenpts_maxerrorseq,\
                                'l2errorseq':l2errorseq,\
                                'lossseq':lossseq,\
                                'lrseq':lrseq,\
                                'maxerrorseq':maxerrorseq,\
                                'resseq':resseq,\
                                }
    if if_true_solution_given() == True:
        if flag_l2error == True:
            data['u_l2norm_x_test'] = u_l2norm_x_test
        if flag_maxerror == True:
            data['u_maxnorm_x_test'] = u_maxnorm_x_test
        if flag_givenpts_l2error == True:
            data['u_l2norm_givenpts'] = u_l2norm_givenpts
        if flag_givenpts_maxerror == True:
            data['u_maxnorm_givenpts'] = u_maxnorm_givenpts       
    
    filename =  path + 'result_d_'+str(d)+'_'+'.data'
    file = open(filename, 'wb')
    pickle.dump(data, file)
    file.close()
    
print('\n Total training running time: ' + str(time))
time_file = open(path + "time.txt", "w")
n = time_file.write(str(time))
time_file.close()  


scipy.io.savemat(path + 'givenpts_l2errorseq.mat', mdict={'givenpts_l2errorseq': givenpts_l2errorseq})
scipy.io.savemat( path + 'testing_l2errorseq.mat', mdict={'l2errorseq': l2errorseq})
scipy.io.savemat(path + 'maxerrorseq.mat', mdict={'maxerrorseq': maxerrorseq})
scipy.io.savemat(path + 'time_seq.mat', mdict={'time_seq': time_seq})

log_file.close()
    

# empty the GPU memory
torch.cuda.empty_cache()
