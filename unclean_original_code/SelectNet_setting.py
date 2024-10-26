# build the neural network to approximate the selection net
import numpy
import torch
from torch import tanh, squeeze, sin, sigmoid, autograd
from torch.nn.functional import relu

class network(torch.nn.Module):
    def __init__(self, d, m, L, activation_type = 'ReLU3', boundary_control_type = 'none', initial_constant = 'none'):
        super(network, self).__init__()
        self.L = L
        self.layer1 = torch.nn.Linear(d,m)
        if L == 2:
            self.layer2 = torch.nn.Linear(m,1)
        elif L == 3:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,1)
        elif L == 4:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,m)
            self.layer4 = torch.nn.Linear(m,1)
        elif L == 5:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,m)
            self.layer4 = torch.nn.Linear(m,m)
            self.layer5 = torch.nn.Linear(m,1)
        elif L == 6:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,m)
            self.layer4 = torch.nn.Linear(m,m)
            self.layer5 = torch.nn.Linear(m,m)
            self.layer6 = torch.nn.Linear(m,1)
        elif L == 7:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,m)
            self.layer4 = torch.nn.Linear(m,m)
            self.layer5 = torch.nn.Linear(m,m)
            self.layer6 = torch.nn.Linear(m,m)
            self.layer7 = torch.nn.Linear(m,1)
        elif L == 10:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,m)
            self.layer4 = torch.nn.Linear(m,m)
            self.layer5 = torch.nn.Linear(m,m)
            self.layer6 = torch.nn.Linear(m,m)
            self.layer7 = torch.nn.Linear(m,m)
            self.layer8 = torch.nn.Linear(m,m)
            self.layer9 = torch.nn.Linear(m,m)
            self.layer10 = torch.nn.Linear(m,1)
        
        if activation_type == 'ReLU3':
            self.activation = lambda x: relu(x**3)
        elif activation_type == 'sigmoid':
            self.activation = lambda x: sigmoid(x)
        elif activation_type == 'tanh':
            self.activation = lambda x: tanh(x)
        elif activation_type == 'sin':
            self.activation = lambda x: sin(x)
        self.boundary_control_type = boundary_control_type
        if boundary_control_type == 'none':
            self.if_boundary_controlled = False
        else:
            self.if_boundary_controlled = True
        if not initial_constant == 'none':
            if L == 2:
                torch.nn.init.constant_(self.layer2.bias, initial_constant) 
            elif L == 3:
                torch.nn.init.constant_(self.layer3.bias, initial_constant) 
            elif L == 4:
                torch.nn.init.constant_(self.layer4.bias, initial_constant) 
            elif L == 5:
                torch.nn.init.constant_(self.layer5.bias, initial_constant) 
            elif L == 6:
                torch.nn.init.constant_(self.layer6.bias, initial_constant) 
            elif L == 7:
                torch.nn.init.constant_(self.layer7.bias, initial_constant) 
            elif L == 10:
                torch.nn.init.constant_(self.layer10.bias, initial_constant) 

    def forward(self, tensor_x_batch):  
        if self.L == 2:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
        elif self.L == 3:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
        elif self.L == 4:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
            y = self.layer4(self.activation(y))
        elif self.L == 5:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
            y = self.layer4(self.activation(y))
            y = self.layer5(self.activation(y))
        elif self.L == 6:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
            y = self.layer4(self.activation(y))
            y = self.layer5(self.activation(y))
            y = self.layer6(self.activation(y))
        elif self.L == 7:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
            y = self.layer4(self.activation(y))
            y = self.layer5(self.activation(y))
            y = self.layer6(self.activation(y))
            y = self.layer7(self.activation(y))
        elif self.L == 10:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
            y = self.layer4(self.activation(y))
            y = self.layer5(self.activation(y))
            y = self.layer6(self.activation(y))
            y = self.layer7(self.activation(y))
            y = self.layer8(self.activation(y))
            y = self.layer9(self.activation(y))
            y = self.layer10(self.activation(y))
        y = y.squeeze(1)
        if self.boundary_control_type == 'none':
            return y
        elif self.boundary_control_type == 'homo_unit_sphere':
            return y*(1-torch.sum(tensor_x_batch**2,1))
        elif self.boundary_control_type == 'homo_unit_cube':
            return y*torch.prod(tensor_x_batch**2-1, 1)


    # to evaluate the solution with numpy array input and output
    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()
    
    # evaluate the second derivative at for k-th coordinate
    def D2_exact(self, tensor_x_batch, k):
        y = self.forward(tensor_x_batch)
        tensor_weight = torch.ones(y.size())
        grad_y = autograd.grad(y, tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True, create_graph=True, only_inputs=True)
        D2y_k = autograd.grad(outputs=grad_y[0][:,k], inputs=tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True)[0][:,k]
        return D2y_k

    # evaluate the Laplace at tensor_x_batch
    def Laplace(self, tensor_x_batch):
        d = tensor_x_batch.shape[1]
        y = self.forward(tensor_x_batch)
        tensor_weight = torch.ones(y.size())
        grad_y = autograd.grad(y, tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True, create_graph=True, only_inputs=True)
        Laplace_y = torch.zeros(y.size())
        for i in range(d):
            Laplace_y = Laplace_y + autograd.grad(outputs=grad_y[0][:,i], inputs=tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True)[0][:,i]
        return Laplace_y

class network_time_depedent(torch.nn.Module):
    def __init__(self, d, m, L, activation_type = 'ReLU3', boundary_control_type = 'none', initial_control_type = 'none', initial_constant = 'none'):
        super(network_time_depedent, self).__init__()
        self.L = L
        self.layer1 = torch.nn.Linear(d+1,m)
        if L == 2:
            self.layer2 = torch.nn.Linear(m,1)
        elif L == 3:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,1)
        elif L == 4:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,m)
            self.layer4 = torch.nn.Linear(m,1)
        elif L == 5:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,m)
            self.layer4 = torch.nn.Linear(m,m)
            self.layer5 = torch.nn.Linear(m,1)
        elif L == 6:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,m)
            self.layer4 = torch.nn.Linear(m,m)
            self.layer5 = torch.nn.Linear(m,m)
            self.layer6 = torch.nn.Linear(m,1)
        elif L == 7:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,m)
            self.layer4 = torch.nn.Linear(m,m)
            self.layer5 = torch.nn.Linear(m,m)
            self.layer6 = torch.nn.Linear(m,m)
            self.layer7 = torch.nn.Linear(m,1)
        elif L == 10:
            self.layer2 = torch.nn.Linear(m,m)
            self.layer3 = torch.nn.Linear(m,m)
            self.layer4 = torch.nn.Linear(m,m)
            self.layer5 = torch.nn.Linear(m,m)
            self.layer6 = torch.nn.Linear(m,m)
            self.layer7 = torch.nn.Linear(m,m)
            self.layer8 = torch.nn.Linear(m,m)
            self.layer9 = torch.nn.Linear(m,m)
            self.layer10 = torch.nn.Linear(m,1)
        
        if activation_type == 'ReLU3':
            self.activation = lambda x: relu(x**3)
        elif activation_type == 'sigmoid':
            self.activation = lambda x: sigmoid(x)
        elif activation_type == 'tanh':
            self.activation = lambda x: tanh(x)
        elif activation_type == 'sin':
            self.activation = lambda x: sin(x)
        self.boundary_control_type = boundary_control_type
        self.initial_control_type = initial_control_type
        if boundary_control_type == 'none':
            self.if_boundary_controlled = False
        else:
            self.if_boundary_controlled = True
        if initial_control_type == 'none':
            self.if_initial_controlled = False
        else:
            self.if_initial_controlled = True
        if not initial_constant == 'none':
            if L == 2:
                torch.nn.init.constant_(self.layer2.bias, initial_constant) 
            elif L == 3:
                torch.nn.init.constant_(self.layer3.bias, initial_constant) 
            elif L == 4:
                torch.nn.init.constant_(self.layer4.bias, initial_constant) 
            elif L == 5:
                torch.nn.init.constant_(self.layer5.bias, initial_constant) 
            elif L == 6:
                torch.nn.init.constant_(self.layer6.bias, initial_constant) 
            elif L == 7:
                torch.nn.init.constant_(self.layer7.bias, initial_constant) 
            elif L == 10:
                torch.nn.init.constant_(self.layer10.bias, initial_constant) 


    def forward(self, tensor_x_batch):  
        if self.L == 2:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
        elif self.L == 3:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
        elif self.L == 4:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
            y = self.layer4(self.activation(y))
        elif self.L == 5:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
            y = self.layer4(self.activation(y))
            y = self.layer5(self.activation(y))
        elif self.L == 6:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
            y = self.layer4(self.activation(y))
            y = self.layer5(self.activation(y))
            y = self.layer6(self.activation(y))
        elif self.L == 7:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
            y = self.layer4(self.activation(y))
            y = self.layer5(self.activation(y))
            y = self.layer6(self.activation(y))
            y = self.layer7(self.activation(y))
        elif self.L == 10:
            y = self.layer1(tensor_x_batch)
            y = self.layer2(self.activation(y))
            y = self.layer3(self.activation(y))
            y = self.layer4(self.activation(y))
            y = self.layer5(self.activation(y))
            y = self.layer6(self.activation(y))
            y = self.layer7(self.activation(y))
            y = self.layer8(self.activation(y))
            y = self.layer9(self.activation(y))
            y = self.layer10(self.activation(y))
        y = y.squeeze(1)
        if self.boundary_control_type == 'homo_unit_cube':
            y = torch.prod(tensor_x_batch[:,1:]**2-1, 1)*y
        elif self.boundary_control_type == 'homo_unit_sphere':
            y = (torch.sum(tensor_x_batch[:,1:]**2, 1)-1)*y    
        if self.initial_control_type == 'homo_parabolic':
            y = tensor_x_batch[:,0]*y
        elif self.initial_control_type == 'homo_hyperbolic':
            y = (tensor_x_batch[:,0]**2)*y
        return y


    # to evaluate the solution with numpy array input and output
    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        tensor_x_batch.requires_grad=False
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()
    
    # evaluate the second derivative at for k-th spatial coordinate
    def D2_exact(self, tensor_x_batch, k):
        y = self.forward(tensor_x_batch)
        tensor_weight = torch.ones(y.size())
        grad_y = autograd.grad(y, tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True, create_graph=True, only_inputs=True)
        D2y_k = autograd.grad(outputs=grad_y[0][:,k+1], inputs=tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True)[0][:,k+1]
        return D2y_k

    # evaluate the Laplace at tensor_x_batch
    def Laplace(self, tensor_x_batch):
        d = tensor_x_batch.shape[1]-1
        y = self.forward(tensor_x_batch)
        tensor_weight = torch.ones(y.size())
        grad_y = autograd.grad(y, tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True, create_graph=True, only_inputs=True)
        Laplace_y = torch.zeros(y.size())
        for i in range(1,d+1):
            Laplace_y = Laplace_y + autograd.grad(outputs=grad_y[0][:,i], inputs=tensor_x_batch, grad_outputs=tensor_weight, retain_graph=True)[0][:,i]
        return Laplace_y

class selection_network(torch.nn.Module):
    def __init__(self, d, m, maxvalue, minvalue, initial_constant = 'none'):
        super(selection_network, self).__init__()
        self.linear1 = torch.nn.Linear(d,m)
        self.linear2 = torch.nn.Linear(m,int(round(m/2)))
        self.linear3 = torch.nn.Linear(int(round(m/2)),1)
        self.maxvalue = maxvalue
        self.minvalue = minvalue
        if not initial_constant == 'none':
            torch.nn.init.constant_(self.linear3.bias, -numpy.log((maxvalue-initial_constant)/(initial_constant-minvalue))) 

    def forward(self, tensor_x_batch):
        y = relu(self.linear1(tensor_x_batch))
        y = relu(self.linear2(y))
        y = sigmoid(self.linear3(y))*(self.maxvalue-self.minvalue)+self.minvalue
        return y.squeeze(1)
    
    # to evaluate the solution with numpy array input and output
    def predict(self, x_batch):
        tensor_x_batch = torch.Tensor(x_batch)
        y = self.forward(tensor_x_batch)
        return y.cpu().detach().numpy()
