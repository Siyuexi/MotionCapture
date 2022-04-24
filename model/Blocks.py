# basic blocks

import torch.nn as nn

# basic block without pooling
class BasicBlock(nn.Module): 
    def __init__(self,channel_num) -> None:
        super().__init__()
        self.channel_num = channel_num

        self.conv = nn.Conv2d(self.channel_num,self.channel_num,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(self.channel_num)
        self.lu = nn.LeakyReLU(0.1)

        # weight init
        nn.init.normal_(self.conv.weight,mean=0,std=1)

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

# transpose convolutional block with batchnorm and leakyrelu
class TransposeBlock(nn.Module):
    def __init__(self,channel_num):
        super().__init__()
        self.channel_num = channel_num

        self.transconv = nn.ConvTranspose2d(self.channel_num,int(self.channel_num/2),kernel_size=4,stride=2)
        self.bn = nn.BatchNorm2d(self.channel_num)
        self.lu = nn.LeakyReLU(0.1) 

    def forward(self,x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.lu(x)
        return x