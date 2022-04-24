# complete model for pre-training in classification task

import torch.nn as nn
from torch import no_grad
from GestaltNet import GestaltNet


class ExtractNet(GestaltNet):
    def __init__(self,img_size,num_backboneblocks=8,joint_params=16,anchor_params=9) -> None:
        super().__init__(img_size,num_backboneblocks,joint_params,anchor_params)
        self.avgpooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512,10)

    def forward(self,x):
        x = self.extractor.forward(x)
        x = self.avgpooling(x)
        x = x.view(x.size(0),-1) # reshape tensor to [batch_size,feature_tensor]
        x = self.fc(x)
        return x


class GeneratNet(GestaltNet):
    def __init__(self,img_size,num_backboneblocks=8,joint_params=16,anchor_params=9) -> None:
        super().__init__(img_size,num_backboneblocks,joint_params,anchor_params)

    def forward(self,x):
        # with no_grad():
        #     x = self.extractor.forward(x)
        x = self.extractor.forward(x)
        prob = self.generator.forward(x)
        return prob


class SegmentNet(GestaltNet):
    def __init__(self,img_size,num_backboneblocks=8,joint_params=16,anchor_params=9) -> None:
        super().__init__(img_size,num_backboneblocks,joint_params,anchor_params)

    def forward(self,x):
        # with no_grad():
        #     x = self.extractor.forward(x)
        x = self.extractor.forward(x)
        loc,sco = self.segmentor.forward(x)
        return loc,sco