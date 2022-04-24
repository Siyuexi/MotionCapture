from torch.utils.data import DataLoader
from utils.Parser import Parser
from utils.tools import anchor_create
from PIL import Image
import matplotlib.pyplot as plt

x = Parser(img_size=512,img_path="D:/MPII_dataset/images")
d = DataLoader(x,batch_size=1,shuffle=True)
a,index = anchor_create(512,8,16)

for id,(data,img_name) in enumerate(d):

    # kan anchor
    img_name=img_name[0]

    print(img_name)
    
    img = Image.open("D:/MPII_dataset/images/"+img_name)
    plt.imshow(img.resize((512,512)))

    ax = plt.gca()
    
    dic = x.label_dict[img_name][1]

    for i in range(len(dic)):
        xmin = dic[i][0]
        ymin = dic[i][1]
        w = dic[i][2]-xmin
        h = dic[i][3]-ymin
        ax.add_patch(plt.Rectangle((xmin,ymin),w,h,color="red",fill=False))

    for i in range(16):
        xmin = a[i][0]
        ymin = a[i][1]
        w = a[i][2] - xmin
        h = a[i][3] - ymin
        ax.add_patch(plt.Rectangle((xmin,ymin),w,h,color="blue",fill=False))

    plt.show()


    # huanyuan
    datas = data.squeeze(0)

    import torch
    image = Image.fromarray(torch.clamp(datas * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
    
    plt.imshow(image)
    plt.show()
    


