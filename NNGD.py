# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:33:54 2023

@author: F520-CJH
"""
from torch import nn
import torch

from torch.nn.parameter import Parameter, UninitializedParameter

class NNGD(nn.Module):

    def __init__(self, M=3, N=4):
        super(NNGD, self).__init__()
        self.M = M
        self.N = N

        self.b = Parameter(torch.empty((1, self.M * self.N)))
        nn.init.trunc_normal_(self.b, std=0.01)
        
    def forward(self, inputs):
        y = self.b
        return y
