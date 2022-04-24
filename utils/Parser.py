# a dataset parser for json

import json
import os
from torch.utils import data
from PIL import Image
import numpy as np



class Parser(data.Dataset):
    def __init__(self,annot_path="dataset/mpii/",img_path="images",img_size=256,type='train') -> None:
        self.annot_path = annot_path 
        if(type == 'train'):
            self.annot_path = self.annot_path + 'train.json'
        elif(type == 'valid'):
            self.annot_path = self.annot_path + "valid.json"
        elif(type== 'test'):
            self.annot_path = self.annot_path + "test.json"

        self.img_path = img_path
        self.img_size = img_size
        
        file = open(self.annot_path, "rb")
        self.annot = json.load(file) 
        file.close()

        self.num_img = self._get_len()
        self.label_dict = self._build_dict()

    def _get_info(self,index,w,h):
        
        joint = np.array(self.annot[index]['joints'])
        joint = np.array([joint[:,0]/w,joint[:,1]/h]).transpose()
        joint = joint*self.img_size
        
        hl = self.annot[index]['scale']*100
        cx = self.annot[index]['center'][0]
        cy = self.annot[index]['center'][1]
        bbox = np.array([(cx-hl)/w,(cy-hl)/h,(cx+hl)/w,(cy+hl)/h])
        bbox = bbox * self.img_size      
        return joint,bbox

    def _get_len(self):
        cnt = 0
        ln = ''
        le = len(self.annot)
        for i in range(le):
            cn = self.annot[i]['image']
            if(ln==cn):
                cnt += 1
            ln = cn
        le = le - cnt
        return le

    def _build_dict(self): # build a label reflection dict

        label_dict = {} # emtpy dict
        index_shift = 0 # zero shift
        for index in range(self.num_img):

            img_name = self.annot[index + index_shift]['image'] # a string

            img = Image.open(self.img_path+'/'+img_name)
            w = img.width
            h = img.height
            
            joint,bbox = self._get_info(index+index_shift,w,h)
        
            joints = [joint]
            bboxes = [bbox]

            if(index+index_shift+1 == len(self.annot)):
                joint,bbox = self._get_info(index+index_shift,w,h)
                joints.append(joint)
                bboxes.append(bbox)
                label_dict[img_name] = [joints,bboxes]
                break
            while(self.annot[index + index_shift + 1]['image']==img_name): # if next img is the same
                
                index_shift += 1
                joint,bbox = self._get_info(index+index_shift,w,h)

                joints.append(joint)
                bboxes.append(bbox)

            label_dict[img_name] = [joints,bboxes]

        return label_dict



    def __getitem__(self,index): # if one image contains n sample, then this image will be trained n times in one epoch.
        img_name = self.annot[index]['image'] 
        img = Image.open(self.img_path+'/'+img_name) 
        img = img.resize((self.img_size,self.img_size))
        img = np.array(img,dtype=np.float32) 
        img = img.transpose((2,0,1))/255
        return img,img_name
    
    def __len__(self):
        return len(self.annot)
