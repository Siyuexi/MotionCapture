# An end2end release model for Android

import torch.nn as nn
import torch
import torchvision
from utils.tools import anchor_create,selective_load
from torch.nn import *
from model.Blocks import *
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

        shift = shift.squeeze(0)
        score = score.squeeze(0)
        heatmap = heatmap.squeeze(0)

        roi = self.proposal_create(self.anchor,shift,nn.functional.softmax(score,dim=1)[:,1],self.img_size)
        joints = torch.zeros([roi.shape[0],self.joint_params,2])
        for i in range(roi.shape[0]):
            xmin = int(roi[i,0])
            ymin = int(roi[i,1])
            xmax = int(roi[i,2])
            ymax = int(roi[i,3])

            view = heatmap[:,xmin:xmax,ymin:ymax].contiguous().view(self.joint_params,-1)
            argmax = torch.argmax(view,dim=1)
            joints[i,:,0] = torch.floor(1.0*argmax/(xmax-xmin))
            joints[i,:,1] = argmax%(ymax-ymin)

        return joints

    def proposal_create(self,anchor,shift,score,img_size,train=False,nms_thresh=0.7,n_train_pre_nms=2000
    ,n_train_post_nms=200, n_test_pre_nms=600, n_test_post_nms=6, min_size=16): # create RoIs for Generator
        
        if train:
            n_pre_nms = n_train_pre_nms
            n_post_nms = n_train_post_nms
        else:
            n_pre_nms = n_test_pre_nms
            n_post_nms = n_test_post_nms

        roi = self.bbox_calculate(anchor, shift)

        roi[:, slice(0, 4, 2)] = torch.clamp(
            roi[:, slice(0, 4, 2)], 0, img_size)
        roi[:, slice(1, 4, 2)] = torch.clamp(
            roi[:, slice(1, 4, 2)], 0, img_size)

        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]

        keep = torch.where((hs >= min_size) & (ws >= min_size))[0]

        roi = roi[keep, :]

        score = score[keep]

        order = torch.argsort(score,descending=True)

        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        keep = torchvision.ops.nms(roi, score, nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi

    def bbox_calculate(self,anchor,shift): # for bbox prediction // type == torch.tensor
        # anchor:[xmin,ymin,xmax,ymax] ==> [xcenter,ycenter,w,h] 
        aw = anchor[:,2] - anchor[:,0]
        ah = anchor[:,3] - anchor[:,1]
        ax = anchor[:,0] + 0.5*aw
        ay = anchor[:,1] + 0.5*ah

        # shift:[dx,dy,dw,dh]
        bx = ax + aw*shift[:,0]
        by = ay + ah*shift[:,1]
        bw = aw*torch.exp(shift[:,2])
        bh = ah*torch.exp(shift[:,3])

        xmin = bx - 0.5*bw
        ymin = by - 0.5*bh
        xmax = bw + xmin
        ymax = bh + ymin

        bbox = torch.stack((xmin,ymin,xmax,ymax)).transpose(0,1)

        return bbox

if __name__ == '__main__':
    model = GestaltNet(256,num_backboneblocks=3,anchor_params=16,joint_params=7)
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)
    model,optimizer = selective_load(model,optimizer,"weights/GestaltNet-joint7-pretrain-epoch-10.pth")
    model.anchor = torch.tensor(model.anchor)
    torch.save(model,"GestaltNet-7-model.pt")

    # model.eval()
    # device = torch.device('cpu')
    # model.to(device)
    # input_tensor = torch.rand(1,3, 128, 128)

    
    # mobile = torch.jit.trace(model, input_tensor)
    # mobile.save("GestaltNet-7.pt")

    # import time

    # toc1 = time.time()
    # x = model(input_tensor)
    # toc2= time.time()
    # print(toc2-toc1)