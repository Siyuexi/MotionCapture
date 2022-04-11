# train

import matplotlib.pyplot as plt
from torchvision import transforms,datasets
from torch import device,cuda,max,nn,optim,save,no_grad,load
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import sys

from model.Extractor import Extractor

num_epochs = 3
batch_size = 16
img_size =256

model_name = "extractor-3"
log = open('log/'+model_name+'.txt','wt')

transform = transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()])

train_set = datasets.CIFAR10(root='dataset',train=True,transform=transform,download=True)
test_set = datasets.CIFAR10(root='dataset',train=False,transform=transform,download=True)

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

def accuracy(predictions, labels):
    
    pred = max(predictions.data, 1)[1] 
    right_num = pred.eq(labels.data.view_as(pred)).sum() 
    return right_num, len(labels)

model = Extractor(img_size)
model = nn.Sequential(model,nn.Linear(512,10))
# print(model)
model = model.to(device)
# model.load_state_dict("weights/"+model_name+'-epoch-'+str(num_epochs)+".pth") # load last state (if any)
model.load_state_dict(load("weights/"+"extractor-2"+'-epoch-'+"2"+".pth"))

best_model_wts = model.state_dict()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

err_record = []
best_acc_r = 1

for epoch in range(num_epochs):

    train_accuracy = [] 
    
    for batch_id, (data,label) in enumerate(train_loader):

        data = data.to(device)
        label = label.to(device)

        model.train()
        
        output =  model(data) 
        loss = criterion(output, label) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        accuracies = accuracy(output, label)
        train_accuracy.append(accuracies)
        
        if batch_id%100 ==0: 

            model.eval() 
            val_accuracy = [] 
            
            for (data, label) in val_loader: 

                data = data.to(device)
                label = label.to(device)

                output = model(data) 

                accuracies = accuracy(output, label) 
                val_accuracy.append(accuracies)
                
            train_r = (sum([tup[0] for tup in train_accuracy]), sum([tup[1] for tup in train_accuracy]))

            val_r = (sum([tup[0] for tup in val_accuracy]), sum([tup[1] for tup in val_accuracy]))
            
            train_acc_r = 100. * train_r[0] / train_r[1]
            val_acc_r = 100. * val_r[0] / val_r[1]
            checkpoint = 'Epoch [{}/{}]\tBatch [{}/{}]\tSample [{}/{}]\tLoss: {:.6f}\tTrainAccuracy: {:.2f}%\tValidationAccuracy: {:.2f}%'.format(
                epoch+1,num_epochs,min(batch_id+100,train_size//batch_size),train_size//batch_size ,min((batch_id+100) * batch_size,train_size), train_size,
                loss.item(), 
                train_acc_r, 
                val_acc_r)
            print(checkpoint,file=log,flush=True)
            print(checkpoint,file=sys.stdout)
            if(val_acc_r > best_acc_r):
                best_acc_r = val_acc_r
                best_model_wts = model.state_dict()
            err_record.append((100 - train_acc_r.cpu(), 100 - val_acc_r.cpu()))
    save(best_model_wts,"weights/"+model_name+'-epoch-'+str(epoch)+".pth")

plt.figure(figsize = (10,7))
plt.plot(err_record)
plt.xlabel('Steps')
plt.ylabel('Error rate(%)')
plt.show()
plt.save("log/"+model_name+".pdf")

save(best_model_wts, "weights/"+model_name+".pth")

model.eval() 
test_accuracy = [] 

with no_grad():
    for data,label in test_loader:
        data = data.to(device)
        label = label.to(device)        
        output = model(data)        
        accuracies = accuracy(output,label)
        test_accuracy.append(accuracies)
        
rights = (sum([tup[0] for tup in test_accuracy]), sum([tup[1] for tup in test_accuracy]))
right_rate = 1.0 * rights[0].detach().to('cpu').numpy() / rights[1]

print("TestAccuracy: ",right_rate,file=log,flush=True)
print("TestAccuracy: ",right_rate,file=sys.stdout)