# some tool functions
from torch import load,save,max
import matplotlib.pyplot as plt

def selective_load(model,optimizer,checkpoint_path,need_optim=False):
    model_state_dict = model.state_dict()
    optim_state_dict = optimizer.state_dict()
    pretrained_pth = load(checkpoint_path)
    pretrained_state_dict = pretrained_pth['state_dict']
    pretrained_optim_dict = pretrained_pth['optimizer']

    for k,v in pretrained_state_dict.items():
        if k in model_state_dict.keys():
            model_state_dict[k] = v
    if need_optim == True:        
        for k,v in pretrained_optim_dict.items():
            if k in optim_state_dict.keys():
                optim_state_dict[k] = v
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optim_state_dict)
    return model,optimizer

def complete_save(model_state_dict,optimizer_state_dict,checkpoint_path):
    save({
        'state_dict':model_state_dict,
        'optimizer':optimizer_state_dict
    },
    checkpoint_path)

def accurate_count(predictions, labels):
    pred = max(predictions.data, 1)[1] 
    right_num = pred.eq(labels.data.view_as(pred)).sum() 
    return right_num, len(labels)

def learning_draw(model_name,error_rate):
    fig = plt.figure()
    plt.figure(figsize = (10,7))
    plt.plot(error_rate)
    plt.xlabel('Steps')
    plt.ylabel('Error rate(%)')
    plt.show()
    fig.savefig("log/"+model_name+".pdf")

def anchor_create():
    pass

def region_split():
    pass

def pose_estimate():
    pass