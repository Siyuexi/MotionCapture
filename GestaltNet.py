# GestaltNet
# forward function has intergrated multiple methods 

import torch.nn as nn
from utils.tools import anchor_create
from model.Extractor import Extractor
from model.Generator import Generator
from model.Segmentor import Segmentor

class GestaltNet(nn.Module):
    def __init__(self,img_size,num_backboneblocks=8,joint_params=16,anchor_params=9) -> None: # num_backboneblocks >= 2
        super().__init__()
        self.img_size = img_size
        self.num_backboneblocks = num_backboneblocks
        self.anchor_params = anchor_params
        self.joint_params = joint_params

        self.anchor,self.legal_index = anchor_create(self.img_size,self.num_backboneblocks,self.anchor_params)

        self.extractor = Extractor(self.num_backboneblocks)
        self.generator = Generator(self.num_backboneblocks,self.joint_params)
        self.segmentor = Segmentor(self.num_backboneblocks,self.anchor_params)
        
    def forward(self,x):
        pass