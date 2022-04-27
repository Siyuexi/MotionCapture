# GestaltNet
# forward function has intergrated multiple methods 

import torch.nn as nn
from torch import tensor
from utils.tools import anchor_create,proposal_create
from model.Extractor import Extractor
from model.Generator import Generator
from model.Segmentor import Segmentor

class GestaltNet(nn.Module):
    def __init__(self,img_size,num_backboneblocks=2,joint_params=16,anchor_params=9) -> None: # num_backboneblocks >= 2
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
        feature = self.extractor(x)
        shift,score = self.segmentor(feature)
        heatmap = self.generator(feature)

        # for i in range(shift.shape[0]): # for training in batchsize != 1 situation
        #     roi = proposal_create(self.anchor,shift[i],nn.functional.softmax(score[i],dim=1)[:,1],self.img_size)
        #     num_roi = roi.shape[0]
        #     heatmap[i] = heatmap[i]*num_roi
        
        roi = proposal_create(self.anchor,shift.squeeze(0),nn.functional.softmax(score.squeeze(0),dim=1)[:,1],self.img_size)
        # print(roi)

        return heatmap.squeeze(0),shift.squeeze(0),score.squeeze(0),roi
