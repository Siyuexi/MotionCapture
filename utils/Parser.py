# a dataset parser for json

import json
from torch.utils import data
from PIL import Image
import numpy as np



class Parser(data.Dataset):
    def __init__(self,annot_path="dataset/mpii/train.json",img_path="D:/MPII_dataset/images",img_size=256) -> None:
        self.annot_path = annot_path
        self.img_path = img_path
        self.img_size = img_size
        
        self.last_name = self.annot[0]['image'] # some people are in the same image, and these samples are adjacent to each other
        self.index_shift = 0

        file = open(annot_path, "rb")
        self.annot = json.load(file) 
        file.close()

    def __getitem__(self,index):
        idx = index + self.index_shift
        img_name = self.annot[idx]['image'] # a string

        joints = []
        bboxes = []

        while(img_name==self.last_name):
            self.last_name = img_name
            self.index_shift += 1 # add shift

            img = Image.open(self.img_path+'/'+img_name)
            w = img.width
            h = img.height
            img.resize((self.img_size,self.img_size))
            img = np.array(img)

            joint = np.array(self.annot[idx]['joints'])
            joint = np.array([joint[:,0]/w,joint[:,1]/h]).transpose()
            joint = joint*self.img_size
            
            hl = self.annot[idx]['scale']*100
            cx = self.annot[idx]['center'][0]
            cy = self.annot[idx]['center'][1]
            bbox = np.array([(cx-hl)/w,(cy-hl)/h,(cx+hl)/w,(cy+hl)/h])
            bbox = bbox * self.img_size

            joints.append(joint)
            bboxes.append(bbox)

        return img,joints,bboxes
    
    def __len__(self):
        cnt = 0
        ln = ''
        le = len(self.annot)
        for i in range(le):
            cn = self.annot[i]['image']
            if(ln==cn):
                cnt += 1
            ln = cn
        le = le - cn
        return le
