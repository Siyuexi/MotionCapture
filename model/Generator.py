# Simple Baseline

import torch.nn as nn
from torch import argmax
from model.Blocks import TransposeBlock

class Generator(nn.Module):
    def __init__(self,num_backboneblocks,joint_params) -> None:
        super().__init__()
        self.num_backboneblocks = num_backboneblocks
        self.joint_params = joint_params

        self.blocks = nn.ModuleList([TransposeBlock(64*(2**(3-i)))for i in range(int(self.num_backboneblocks/2))])
        self.estimation = nn.Conv2d(64,self.joint_params,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.blocks(x)
        x = self.estimation(x)
        n,_,w,h = x.shape
        
        prob = x.view(n,self.joint_params,w*h) # reshape to [batchsize,jointnumber,w*h]
        prob = nn.functional.softmax(prob,dim=2) # calculate probability
        prob = prob.view(n,self.joint_params,w,h)
        return prob
        