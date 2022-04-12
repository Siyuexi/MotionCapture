# Residual Sampling

import torch.nn as nn 

# basic block without pooling
class BasicBlock(nn.Module): 
    def __init__(self,channel_num) -> None:
        super().__init__()
        self.channel_num = channel_num

        self.conv = nn.Conv2d(self.channel_num,self.channel_num,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(self.channel_num)
        self.lu = nn.LeakyReLU(0.1)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.lu(x)
        return x

# residual block with batchnorm and leakyrelu
class ResidualBlock(nn.Module):
    def __init__(self,channel_num,block_num=2) -> None:
        super().__init__()
        self.channel_num = channel_num
        self.block_num = block_num

        self.blocks = nn.ModuleList([BasicBlock(self.channel_num) for i in range(self.block_num)])
        
        
    def forward(self,x):
        t = x
        for i in range(self.block_num):
            x = self.blocks[i](x)
        x = x + t
        return x

# general feature extractor backbone
class Extractor(nn.Module):
    def __init__(self,block_num=8) -> None:
        super().__init__()
        self.block_num=block_num

        self.conv1 = nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(64*(2**int(i/2)))for i in range(self.block_num)])
        self.samplings = nn.ModuleList([nn.Conv2d(64*(2**i),64*(2**(i+1)),kernel_size=3,stride=2,padding=1)for i in range(int(self.block_num/2)-1)])


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        for i in range(int(self.block_num/2)-1):
            x = self.blocks[i*2](x)
            x = self.blocks[i*2+1](x)
            x = self.samplings[i](x)
        x = self.blocks[self.block_num-1](x)
        return x
