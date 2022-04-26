# Region Proposal

import torch.nn as nn
from model.Blocks import BasicBlock


class Segmentor(nn.Module):
    def __init__(self,num_backboneblocks,anchor_params) -> None:
        super().__init__() 
        self.num_backboneblocks = num_backboneblocks
        self.anchor_params = anchor_params

        self.blocks = BasicBlock(64*(2**(self.num_backboneblocks-1)))
        self.classification = nn.Conv2d(64*(2**(self.num_backboneblocks-1)),2*self.anchor_params,kernel_size=1,stride=1,padding=0) # foreground or background
        self.regression = nn.Conv2d(64*(2**(self.num_backboneblocks-1)),4*self.anchor_params,kernel_size=1,stride=1,padding=0) # bbox shift of anchor

        # weight init
        nn.init.normal_(self.classification.weight,mean=0,std=1)
        nn.init.normal_(self.regression.weight,mean=0,std=1)

    def forward(self,x):
        x = self.blocks(x)
        n,_,h,w = x.shape
                
        loc = self.regression(x) # shape : [batchsize,4*anchornumber,w,h]
        loc = loc.permute(0,2,3,1).contiguous().view(n,-1,4) # reshape to [batchsize,totalpixelsnumber,4]
        
        sco = self.classification(x) # shape : [batchsize,2*anchornumber,w,h]
        sco = sco.permute(0,2,3,1).contiguous().view(n, -1, 2)# reshape to [batchsize,totalpixelsnumber,2]
        return loc,sco

    

