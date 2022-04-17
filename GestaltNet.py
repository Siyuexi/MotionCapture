# GestaltNet
# forward function has intergrated multiple methods 

import torch.nn as nn
from utils.tools import pose_estimate,anchor_create,region_split
from model.Extractor import Extractor
from model.Generator import Generator
from model.Segmentor import Segmentor

class GestaltNet(nn.Module):
    def __init__(self,num_backboneblocks=8,joint_params=16,anchor_params=9) -> None: # num_backboneblocks >= 2
        super().__init__()
        self.num_backboneblocks = num_backboneblocks
        self.anchor_params = anchor_params
        self.joint_params = joint_params

        self.extractor = Extractor(self.num_backboneblocks)
        self.generator = Generator(self.num_backboneblocks,self.joint_params)
        self.segmentor = Segmentor(self.num_backboneblocks,self.anchor_params)
        
    def forward(self,x):
        pass