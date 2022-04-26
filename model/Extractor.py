# Residual Sampling

import torch.nn as nn 
from model.Blocks import ResidualBlock


# general feature extractor backbone
class Extractor(nn.Module):
    def __init__(self,block_num) -> None:
        super().__init__()
        self.block_num=block_num

        self.conv = nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(64*(2**int(i/2)))for i in range(self.block_num)])
        self.samplings = nn.ModuleList([nn.Conv2d(64*(2**i),64*(2**(i+1)),kernel_size=3,stride=2,padding=1)for i in range(self.block_num-1)])

        # weight init
        nn.init.normal_(self.conv.weight,mean=0,std=1)

    def forward(self,x):
        x = self.conv(x)
        for i in range(int(self.block_num/2)-1):
            x = self.blocks[i*2](x)
            x = self.blocks[i*2+1](x)
            x = self.samplings[i](x)
        x = self.blocks[self.block_num-2](x) # x means feature
        x = self.blocks[self.block_num-1](x)
        return x
