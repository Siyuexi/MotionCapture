# some tool functions

from torch import load,save,max,min, tensor,zeros,is_tensor,from_numpy,exp,stack
from torchvision.ops import nms
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    fig = plt.figure(figsize = (10,7))
    plt.plot(error_rate)
    plt.xlabel('Steps')
    plt.ylabel('Error rate(%)')
    plt.show()
    fig.savefig("log/"+model_name+".pdf")

def unmap(data,count,index,fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count,)+data.shape[1:],dtype=data.dtype)
        ret.fill(fill)
        ret[index,:]=data
    return ret

def shift_calculate(anchor,gt): # for self-defined pre-training on Segmentor // type == numpy.array
    # anchor:[xmin,ymin,xmax,ymax] ==> [xcenter,ycenter,w,h] 
    aw = anchor[:,2] - anchor[:,0]
    ah = anchor[:,3] - anchor[:,1]
    ax = anchor[:,0] + 0.5*aw
    ay = anchor[:,1] + 0.5*ah

    # gt:[xmin,ymin,xmax,ymax] ==> [x,y,w,h]
    tw = gt[:,2] - gt[:,0]
    th = gt[:,3] - gt[:,1]
    tx = gt[:,0] + 0.5*tw
    ty = gt[:,1] + 0.5*th 

    # get a very small number eps to avoid "/0" exception
    eps = np.finfo(ah.dtype).eps
    h = np.maximum(ah,eps)
    w = np.maximum(aw,eps)

    dy = (ty-ay)/h
    dx = (tx-ax)/w
    dh = np.log(th/h)
    dw = np.log(tw/w)

    shift = np.vstack((dx,dy,dw,dh)).transpose()

    return shift

def bbox_calculate(anchor,shift): # for bbox prediction // type == torch.tensor
    # anchor:[xmin,ymin,xmax,ymax] ==> [xcenter,ycenter,w,h] 
    aw = anchor[:,2] - anchor[:,0]
    ah = anchor[:,3] - anchor[:,1]
    ax = anchor[:,0] + 0.5*aw
    ay = anchor[:,1] + 0.5*ah

    # shift:[dx,dy,dw,dh]
    bx = ax + aw*shift[:,0].cpu().detach().numpy()
    by = ay + ah*shift[:,1].cpu().detach().numpy()
    bw = aw*np.exp(shift[:,2].cpu().detach().numpy())
    bh = ah*np.exp(shift[:,3].cpu().detach().numpy())

    xmin = bx - 0.5*bw
    ymin = by - 0.5*bh
    xmax = bw + xmin
    ymax = bh + ymin

    bbox = np.vstack((xmin,ymin,xmax,ymax)).transpose()

    return bbox

def IoU_calculate(box1,box2,threshold=0.5,type=0): # type==0:find those who >= threshold; type==1:find those who <=threshold;else do not calculate acc
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    inter_rect_x1 = np.max([b1_x1, b2_x1])
    inter_rect_y1 = np.max([b1_y1, b2_y1])
    inter_rect_x2 = np.min([b1_x2, b2_x2])
    inter_rect_y2 = np.min([b1_y2, b2_y2])

    if(inter_rect_x1>=inter_rect_x2 or inter_rect_y1>=inter_rect_y2):
        iou = 0

    else:

        inter_area = (inter_rect_x2 - inter_rect_x1) * (inter_rect_y2 - inter_rect_y1)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / (b1_area + b2_area - inter_area)

    if(type==0):
        num = np.sum(i >= threshold for i in iou)
    elif(type==1):
        num = np.sum(i <= threshold for i in iou)
    else:
        return iou

    return iou,num

def anchor_create(img_size,num_backboneblocks,anchor_params): # anchor_params == 9 or 16
    # type of anchor coordinates : Numpy (not Tensor)
    
    def _generate_base():   #  generate a basic anchor (and all other anchors can be generated from the shift of the base)
        base_size = 2**int(num_backboneblocks) # 1 feature pixel = 'base_size' origin pixels
        if anchor_params == 9:
            scales = [8,16,36]
            ratios = [0.5,1,2]
        elif anchor_params == 16:
            scales = [8,16,24,32]
            ratios = [0.5,0.707,1.414,2]
        else:
            raise Exception('anchor_params can only be 9 or 16 currently.')
        y = base_size/2
        x = base_size/2
        base_anchor = np.zeros((len(ratios) * len(scales), 4), dtype=np.float32) # shape : [anchor_types,4coordinates]
        for i in range(len(ratios)):
            for j in range(len(scales)):
                h = base_size * scales[j] * np.sqrt(ratios[i])
                w = base_size * scales[j] * np.sqrt(1. / ratios[i])
                index = i * len(scales) + j
                
                # anchor coordinates must be integer(or not? the shifts are not integer.)
                # base_anchor[index, 0] = int(y - h / 2)
                # base_anchor[index, 1] = int(x - w / 2)
                # base_anchor[index, 2] = int(y + h / 2)
                # base_anchor[index, 3] = int(x + w / 2)
                base_anchor[index, 0] = x - w / 2
                base_anchor[index, 1] = y - h / 2
                base_anchor[index, 2] = x + w / 2
                base_anchor[index, 3] = y + h / 2

        return base_anchor

    def _generate_shift(): # ...on origin image scale
        stride = 2**int(num_backboneblocks) 
        shift_y = np.arange(0,img_size,stride)
        shift_x = np.arange(0,img_size,stride)
        # shift_x and shift y all equal to [0,16,32,...]
        shift_x,shift_y = np.meshgrid(shift_x,shift_y)
        '''
        shift_x = [[0,16,32,...],
                   [0,16,32,...],
                   [0,16,32,...],
                   ...]
        shift_y = [[0, 0, 0,... ],
                   [16,16,16,...],
                   [32,32,32,...],
                   ...]
        '''
        shift = np.stack((shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel()), axis=1)
        '''
        shift_x.ravel() : [0,16,32,...,0,16,32,..,0,16,32,...],(1,w*h)
        shift_y.ravel() : [0,0,0,...,16,16,16,...,32,32,32,...],(1,w*h)
        shift : 
            [[0,  0, 0,  0],
            [16, 0, 16, 0],
            [32, 0, 32, 0],
            ...]
        '''
        
        return shift

    base_anchor = _generate_base()
    shift = _generate_shift() 
    num_anchor_per_loc = base_anchor.shape[0]
    num_loc = shift.shape[0]

    # "Numpy Broadcasting:"
    anchor = base_anchor.reshape((1,num_anchor_per_loc,4))+shift.reshape((1,num_loc,4)).transpose((1,0,2)) # after broadcasting, shape : (num_loc,num_anchor_per_loc,4)
    anchor = anchor.reshape((num_loc*num_anchor_per_loc,4)).astype(np.float32) # reshape anchor to : (num_loc*num_anchor_per_loc,4)
    # get legal index inside the image
    index = np.where(
        (anchor[:,0]>=0) &
        (anchor[:,1]>=0) &
        (anchor[:,2]<=img_size) &
        (anchor[:,3]<=img_size) 
    )[0]

    return anchor,index

def sample_create(anchor,index,gt,num_sample=256,posi_thresh=0.7,nega_thresh=0.3,ratio=0.5): # create samples for Segmentor
    
    def _create_ious():
        legal_anchor = anchor[index]
        len_anchor = len(legal_anchor)
        len_bbox = len(gt)
        ious = np.empty((len_anchor,len_bbox)) 
        for i in range(len_anchor):
            for j in range(len_bbox):
                ious[i,j] = IoU_calculate(legal_anchor[i],gt[j],type=-1)

        argmax_ious = ious.argmax(axis=1) # shape : [1,len_bbox] . For every anchor find bbox whose ious is the biggest
        max_ious = ious[np.arange(len_anchor), argmax_ious]

        gt_argmax_ious = ious.argmax(axis=0) # shape : [len_anchor,1]. For every gt_bbox find anchor whose ious is the biggest
        gt_max_ious = ious[gt_argmax_ious, np.arange(len_bbox)]
        
        gt_argmax_ious = np.where(ious == gt_max_ious)[0] # just a reshape:[len_anchor] (can be seen as [1,len_anchor])

        # if(len(gt_argmax_ious)==0):
        #     # print(gt_argmax_ious)
        #     # print( gt_max_ious)
        #     # print(ious.argmax(axis=0))
        #     # print(np.arange(len_bbox))
        #     # print(len_bbox)
        # print(ious)
        # print(len(gt))
        # print(len(legal_anchor))
        # print(legal_anchor)
        # print(gt)
        # exit()


        return argmax_ious,max_ious,gt_argmax_ious 

    def _create_label():
        label = np.empty((len(index),),dtype=np.int64)
        label.fill(-1)
        argmax_ious,max_ious,gt_argmax_ious = _create_ious()

        label[max_ious<nega_thresh] = 0 # tag all negative sample
        label[gt_argmax_ious] = 1 # tag sample who has the biggest iou as positive
        label[max_ious >= posi_thresh] = 1 # tag the remain positive sample

        # sampling
        n_pos = int(ratio * num_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        n_neg = np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    argmax_ious, label = _create_label()
    shift = shift_calculate(anchor[index],np.array(gt,dtype=np.float32)[argmax_ious])
    label = unmap(label,len(anchor),index,fill=-1)
    shift = unmap(shift,len(anchor),index,fill=0)
    
    return shift, label

def proposal_create(anchor,shift,score,img_size,train=False,nms_thresh=0.7,n_train_pre_nms=12000
    ,n_train_post_nms=2000, n_test_pre_nms=6000, n_test_post_nms=300, min_size=16): # create RoIs for Generator
        
        if train:
            n_pre_nms = n_train_pre_nms
            n_post_nms = n_train_post_nms
        else:
            n_pre_nms = n_test_pre_nms
            n_post_nms = n_test_post_nms

        roi = bbox_calculate(anchor, shift)
        # print(roi)
        # print("roi:"+str(roi.shape))

        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size)
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size)
        # print(roi)
        # print(roi.shape)

        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]

        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        # print(keep)
        # print(keep.shape)

        roi = roi[keep, :]
        # print(roi)
        # print(roi.shape)
        # print(score)
        # print(score.shape)

        score = score[keep]

        # print('score'+str(score))
        # print(score.shape)

        score = score.cpu().detach().numpy()
        order = score.ravel().argsort()[::-1]

        # print(order)
        # print(order.shape)

        if n_pre_nms > 0:
            order = order[:n_pre_nms]

        # print(order)
        # print(order.shape)

        roi = roi[order, :]
        score = score[order]

        keep = nms(from_numpy(roi), from_numpy(score), nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi

def target_create(roi,gt):

    len_roi = roi.shape[0]
    len_bbox = len(gt)  
    ious = np.empty((len_roi,len_bbox))  
    for i in range(len_roi):
        for j in range(len_bbox):
            ious[i,j] = IoU_calculate(roi[i],gt[j],type=-1)

    roi_index = ious.argmax(axis=0) # for every gt, find the biggest iou roi's index
    target = roi[roi_index]

    return target

def label_create(joint,img_size,joint_params,sigma=16): #
    label = np.empty([len(joint),joint_params,img_size,img_size])
    # print(label.shape)
    for i in range(len(joint)):
        for j in range(joint_params):
            x1 = np.linspace(1,img_size,img_size)
            y1 = np.linspace(1,img_size,img_size)
            [x,y] = np.meshgrid(x1,y1)
            x = x - joint[i][j][0]
            y = y - joint[i][j][1]
            d2 = x*x + y*y
            e2 = 2.0*sigma*sigma
            exponent = d2/e2
            label[i,j,:,:] = np.exp(-exponent)

    # # label testing
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")

    # x = np.arange(0,img_size)
    # y = np.arange(0,img_size)
    # x,y = np.meshgrid(x, y)


    # ax.plot_surface(x,y,label[0][0])
    # ax.set_xlabel("X Label")
    # ax.set_ylabel("Y Label")
    # ax.set_zlabel("Z Label")

    # ax.set_title("3D surface plot")
    # plt.show()
    # print(np.max(label[0][0]))
    # print(np.argmax(label[0][0]))
    # print(label[0][0])

    return label

def criterion_key(pred,label,target):
    loss = 0
    for i in range(len(label)):
        xmin = int(target[i,0])
        ymin = int(target[i,1])
        xmax = int(target[i,2])
        ymax = int(target[i,3])

        sub = torch.tensor(label[i,xmin:xmax,ymin:ymax])-pred[xmin:xmax,ymin:ymax]
        loss = loss + torch.sum(torch.mul(sub,sub))
        # print(sub)
    return loss


def poses_draw(imgs, num_rows, num_cols, titles=None, scale=1.5):  

    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if is_tensor(img):
            ax.imshow(img.numpy() * 255)
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def bboxes_draw(axes, bboxes, labels=None, colors=None):
    
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    def _bbox_to_rect(bbox, color):
        return plt.Rectangle(
            xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
            fill=False, edgecolor=color, linewidth=2)

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = _bbox_to_rect(bbox.numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))