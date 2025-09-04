import sys
from torch.nn import Linear
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
import math
from torch.utils.data import (DataLoader, Dataset)
torch.set_default_tensor_type(torch.DoubleTensor)
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import torch
torch.set_default_tensor_type(torch.FloatTensor)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Current device:", device)



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=None):  # 原来是true
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj,active=True):
        input = input.float()
        support = torch.mm(input, self.weight)
        # Debugging information
        # print(f"input shape: {input.shape}")
        # print(f"adj shape: {adj.shape}")
        # print(f"support shape: {support.shape}")
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
