# Simple Baseline

import torch.nn as nn
from model.Blocks import TransposeBlock

class Generator(nn.Module):
    def __init__(self,num_backboneblocks,joint_params) -> None:
        super().__init__()
        self.num_backboneblocks = num_backboneblocks
        self.joint_params = joint_params

        self.blocks = nn.ModuleList([TransposeBlock(64*(2**(3-i)))for i in range(int(self.num_backboneblocks/2))])
        self.estimation = nn.Conv2d(64,sum(self.joint_params),kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        pass
        