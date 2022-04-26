# Simple Baseline

import torch.nn as nn
from torch import argmax
from model.Blocks import TransposeBlock

class Generator(nn.Module):
    def __init__(self,num_backboneblocks,joint_params) -> None:
        super().__init__()
        self.num_backboneblocks = num_backboneblocks
        self.joint_params = joint_params

        self.blocks = nn.ModuleList([TransposeBlock(64*(2**(int(self.num_backboneblocks/2-1)-i)))for i in range(int(self.num_backboneblocks/2))])
        self.estimation = nn.Conv2d(32,self.joint_params,kernel_size=1,stride=1,padding=0)
        
        # weight init
        nn.init.normal_(self.estimation.weight,mean=0,std=1)

    def forward(self,x):
        for i in range(int(self.num_backboneblocks/2)):
            x = self.blocks[i](x)
        x = self.estimation(x)
        n,_,w,h = x.shape
        
        prob = x.view(n,self.joint_params,w*h) # reshape to [batchsize,jointnumber,w*h]
        prob = nn.functional.softmax(prob,dim=2) # calculate probability
        prob = prob.view(n,self.joint_params,w,h)
        return prob
        