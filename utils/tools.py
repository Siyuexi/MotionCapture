# some tool functions
from torch import load,save,max,min,zeros,is_tensor
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
    fig = plt.figure(figsize = (10,7))
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

def IoU_calculate(box1,box2,threshold=0.5,device='cuda'): # device could not be cuda on Android 
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)

    inter_area = max(inter_rect_x2 - inter_rect_x1 + 1, zeros(inter_rect_x2.shape).to(device)) * max(
        inter_rect_y2 - inter_rect_y1 + 1, zeros(inter_rect_x2.shape).to(device))

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)
    
    right_num = sum(i >= threshold for i in iou)

    return right_num, len(iou)

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