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

        file = open(annot_path, "rb")
        self.annot = json.load(file) 
        file.close()

    def __getitem__(self,index):
        img_name = self.annot[index]['image'] # a string

        img = Image.open(self.img_path+'/'+img_name)
        w = img.width
        h = img.height
        img.resize((self.img_size,self.img_size))
        img = np.array(img)

        joints = np.array(self.annot[index]['joints'])
        joints = np.array([joints[:,0]/w,joints[:,1]/h]).transpose()
        joints = joints*self.img_size
        
        hl = self.annot[index]['scale']*100
        cx = self.annot[index]['center'][0]
        cy = self.annot[index]['center'][1]
        bbox = np.array([(cx-hl)/w,(cy-hl)/h,(cx+hl)/w,(cy+hl)/h])
        bbox = bbox * self.img_size

        return img,joints,bbox
    
    def __len__(self):
        return len(self.annot)
