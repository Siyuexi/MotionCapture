# Region Proposal

import torch.nn as nn
from model.Blocks import BasicBlock


class Segmentor(nn.Module):
    def __init__(self,num_backboneblocks,anchor_params) -> None:
        super().__init__() 
        self.num_backboneblocks = num_backboneblocks
        self.anchor_params = anchor_params

        self.blocks = BasicBlock(64*(2**(int(self.num_backboneblocks/2-1))))
        self.classification = nn.Conv2d(64*(2**(int(self.num_backboneblocks/2-1))),2*self.anchor_params[0]*self.anchor_params[1],kernel_size=1,stride=1,padding=0)
        self.regression = nn.Conv2d(64*(2**(int(self.num_backboneblocks/2-1))),4*self.anchor_params[0]*self.anchor_params[1],kernel_size=1,stride=1,padding=1)

    def forward(self,x):
        pass



    

