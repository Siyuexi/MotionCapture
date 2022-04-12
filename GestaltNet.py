# GestaltNet
# forward function has intergrated multiple methods 

import torch.nn as nn
from utils.tools import pose_estimate,anchor_create,region_split
from model.Extractor import Extractor
from model.Generator import Generator
from model.Segmentor import Segmentor

class GestaltNet(nn.Module):
    def __init__(self,img_size,num_backboneblocks=8) -> None:
        super().__init__()
        self.img_size = img_size
        self.extractor = Extractor(self.img_size,num_backboneblocks)
        self.generator = Generator()
        self.segmentor = Segmentor()
        
    def forward(self,x):
        pass