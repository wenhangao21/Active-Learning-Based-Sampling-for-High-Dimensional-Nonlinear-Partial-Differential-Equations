import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd, tanh
from torch.nn.functional import relu


class Residual_Block(nn.Module):
    def __init__(self, m):
        super(Residual_Block, self).__init__()
        # create the necessary linear layers
        self.L1 = nn.Linear(m, m)
        self.L2 = nn.Linear(m, m)
        # choose appropriate activation function
        self.phi = nn.Tanh()

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x)))) + x


class drnn(nn.Module):
    def __init__(self,  m):
        super(drnn, self).__init__()
        # set parameters
        self.m = m
        self.phi = nn.Tanh()
        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(2, m))

        # add middle blocks to list
        for i in range(4):
            self.stack.append(Residual_Block(m))

        # add output linear layer
        self.stack.append(nn.Linear(m, 1))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x


class basicnn(torch.nn.Module):
    def __init__(self, m):
        super(basicnn, self).__init__()
        self.layer1 = nn.Linear(2, m)
        self.layer2 = nn.Linear(m, m)
        """
        self.layer3 = nn.Linear(m, m)
        self.layer4 = nn.Linear(m, m)
        self.layer5 = nn.Linear(m, m)
        self.layer6 = nn.Linear(m, m)
        self.layer7 = nn.Linear(m, m)
        self.layer8 = nn.Linear(m, m)
        self.layer9 = nn.Linear(m, m)
        """
        self.layer10 = nn.Linear(m, 1)
        self.activation = lambda x: tanh(x)

    def forward(self, tensor_x_batch):
        y = self.layer1(tensor_x_batch)
        y = self.layer2(self.activation(y))
        """
        y = self.layer3(self.activation(y))
        y = self.layer4(self.activation(y))
        y = self.layer5(self.activation(y))
        y = self.layer6(self.activation(y))
        y = self.layer7(self.activation(y))
        y = self.layer8(self.activation(y))
        y = self.layer9(self.activation(y))
        """
        y = self.layer10(self.activation(y))
        y = y.squeeze(0)
        return y


