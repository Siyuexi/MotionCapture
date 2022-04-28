# train

import time
from torch import device,cuda,nn,optim,no_grad,tensor,set_printoptions
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import sys
from numpy import where

from GestaltNet import  GestaltNet
from utils.tools import IoU_calculate, accurate_count, label_create,selective_load,complete_save,learning_draw,anchor_create,sample_create,target_create,criterion_key
from utils.Parser import Parser

joint_list = [0,5,6,7,9,10,15] # for 7 joint training
# joint_list = [0,2,3,5,12,13,9,10,15] # for 9 joint training

num_epochs = 16
batch_size = 1 # for rpn training, batch_size fixed at 1
img_size = 128

print_iter_loss = 200 # after 'print_iter_loss' batch print a loss log
print_iter_acc = 2 # after 'print_iter_acc' epoch print a acc log
 
num_sample = 10 # rpn sample number
num_backboneblock = 2 
num_anchor = 9
num_joint = 7
lambda_cls = 1 # weight of classification loss
lambda_loc = 1 # weight of localization loss
lambda_key = 1 # weight of keypoint loss

model_name = "GestaltNet"
log = open('log/'+model_name+'.txt','wt')

print("loading training dataset")
train_set = Parser(img_size=img_size,img_path="D:/MPII_dataset/images",type='valid') # use smaller set
print("loading testing dataset")
test_set = Parser(img_size=img_size,img_path="D:/MPII_dataset/images",type='valid')

test_size = len(test_set)
indices = range(test_size)
indices_val = indices[:int(test_size/2)]
indices_test = indices[int(test_size/2):]  
sampler_val = SubsetRandomSampler(indices_val)
sampler_test = SubsetRandomSampler(indices_test)

train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(dataset=test_set,batch_size = batch_size,sampler = sampler_val)
test_loader = DataLoader(dataset=test_set,batch_size = batch_size,sampler = sampler_test)

train_size =  len(train_loader.dataset)
print('shape of train set:',train_size,file=log,flush=True)
print('shape of train set:',train_size,file=sys.stdout)
print('shape of test set:',test_size,file=log,flush=True)
print('shape of test set:',test_size,file=sys.stdout)

device = device("cuda" if cuda.is_available() else "cpu")
print("device : "+str(device),file=log,flush=True)
print("device : "+str(device),file=sys.stdout)

model = GestaltNet(img_size,num_backboneblocks=num_backboneblock,anchor_params=num_anchor,joint_params=num_joint)
# print(model)
model = model.to(device)

best_model_wts = model.state_dict()
criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
criterion_loc = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)
# optimizer = optim.Adam(model.parameters(),lr=5e-4)
schedule = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda iter: 0.9*iter)

# model,optimizer = selective_load(model,optimizer,"weights/segmentor-batchnorm-epoch-0.pth")
 #model,optimizer = selective_load(model,optimizer,"weights/"+model_name+'-epoch-'+"4"+".pth")

err_record = []
best_acc_r = 1

anchor,anchor_index = anchor_create(img_size,num_backboneblocks=num_backboneblock,anchor_params=num_anchor)

for epoch in range(num_epochs):

    train_accuracy = []
    num_legal_sample = 0
    
    for batch_id, (data,img_name) in enumerate(train_loader):

        # bacht size can only be 1 in rpn training
        data = data.to(device)
        joints = train_set.label_dict[img_name[0]][0]
        bboxes = train_set.label_dict[img_name[0]][1]
        
        if(len(joints[0])!=16): # strip abnormal sample
            continue
            
        for i in range(len(joints)):
            joints[i] = joints[i][joint_list]

        model.train()

        bbox_bias,bbox_label = sample_create(anchor,anchor_index,bboxes,num_sample=num_sample,posi_thresh=0.7,nega_thresh=0.3)

        pos_index = where(bbox_label==1)[0]
        num_legal_sample = num_legal_sample + 2*len(pos_index)

        heatmap,shift,score,roi =  model(data)

        target = target_create(roi,bboxes)

        joint_label = label_create(joints,img_size,num_joint)

        bbox_bias = tensor(bbox_bias).to(device)
        bbox_label = tensor(bbox_label).to(device)

        loss_cls = criterion_cls(score, bbox_label) 
        loss_loc = criterion_loc(shift[pos_index], bbox_bias[pos_index])
        loss_key = criterion_key(heatmap,joint_label,roi,target,img_size,device=device)
        loss = (lambda_cls*loss_cls + lambda_loc*loss_loc)/len(pos_index) + (lambda_key*loss_key)/len(target) # loss batch norm because every image the sample number(aka ture batchsize) is unknown

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        accuracies = accurate_count(score, bbox_label)
        train_accuracy.append(accuracies) 
        
        if batch_id%print_iter_loss ==0: 
            
            checkpoint = 'Epoch [{}/{}]\tBatch [{}/{}]\tSample [{}/{}]\tClsLoss: {:.6f}\tLocLoss: {:.6f}\tKeyLoss: {:.6f}'.format(
                epoch+1,num_epochs,
                min(batch_id+print_iter_loss,train_size//batch_size),train_size//batch_size,
                min((batch_id+print_iter_loss) * batch_size,train_size), train_size,
                lambda_cls*loss_cls.item(),
                lambda_loc*loss_loc.item(),
                lambda_key*loss_key.item()
                )
            print(checkpoint,file=log,flush=True)
            print(checkpoint,file=sys.stdout)

    if epoch%print_iter_acc ==0:
        model.eval() 
        val_accuracy = []
        num_legal_sample_v = 0
        
        for (data, img_name) in val_loader: 

            data = data.to(device)
            bboxes = test_set.label_dict[img_name[0]][1]

            _,label = sample_create(anchor,anchor_index,bboxes,num_sample=num_sample)
            pos_index_v = where(label==1)[0]
            num_legal_sample_v = num_legal_sample_v + 2*len(pos_index_v)
            
            _,_,score,_ =  model(data)

            label = tensor(label).to(device)

            accuracies = accurate_count(score, label) 
            val_accuracy.append(accuracies)
            
        train_r = (sum([tup[0] for tup in train_accuracy]), num_legal_sample)

        val_r = (sum([tup[0] for tup in val_accuracy]), num_legal_sample_v)
        
        train_acc_r = 100. * train_r[0] / train_r[1]
        val_acc_r = 100. * val_r[0] / val_r[1]
        checkpoint = 'Epoch [{}/{}]\tTrainAccuracy: {:.2f}%\tValidationAccuracy: {:.2f}%'.format(
            epoch+1, num_epochs,
            train_acc_r, # rpn training dosen't care about loc acc because positive label number is always more than ground truth number
            val_acc_r)
        print(checkpoint,file=log,flush=True)
        print(checkpoint,file=sys.stdout)
        if(val_acc_r > best_acc_r):
            best_acc_r = val_acc_r
            best_model_wts = model.state_dict()
        err_record.append((100 - train_acc_r.cpu(), 100 - val_acc_r.cpu()))

        train_accuracy = [] # clean the history
        num_legal_sample = 0
        num_legal_sample_v = 0

    schedule.step()

    complete_save(best_model_wts,optimizer.state_dict(),"weights/"+model_name+'-epoch-'+str(epoch)+".pth")

learning_draw(model_name,err_record)

# model.eval() 
# test_accuracy = [] 

# with no_grad():
#     for data,label in test_loader:
#         data = data.to(device)
#         label = label.to(device)        
#         output = model(data)        
#         accuracies = accurate_count(output,label)
#         test_accuracy.append(accuracies)
        
# rights = (sum([tup[0] for tup in test_accuracy]), sum([tup[1] for tup in test_accuracy]))
# right_rate = 1.0 * rights[0].detach().to('cpu').numpy() / rights[1]

# print("TestAccuracy: ",right_rate,file=log,flush=True)
# print("TestAccuracy: ",right_rate,file=sys.stdout)