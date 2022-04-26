from torch.utils.data import DataLoader
from utils.Parser import Parser
from utils.tools import anchor_create
from PIL import Image
import matplotlib.pyplot as plt

img_size = 256
x = Parser(img_size=img_size,img_path="D:/MPII_dataset/images")
print('done')
d = DataLoader(x,batch_size=1,shuffle=True)
print('done')
a,index = anchor_create(img_size,8,16)

for id,(data,img_name) in enumerate(d):

    # kan anchor
    img_name=img_name[0]

    print(img_name)
    
    img = Image.open("D:/MPII_dataset/images/"+img_name)
    plt.imshow(img.resize((img_size,img_size)))

    ax = plt.gca()
    
    dic1 = x.label_dict[img_name][0]
    dic2 = x.label_dict[img_name][1]

    for i in range(len(dic1)):
        for j in range(16):
            plt.scatter(dic1[i][j][0],dic1[i][j][1],s=25,color='green')

    for i in range(len(dic2)):
        xmin = dic2[i][0]
        ymin = dic2[i][1]
        w = dic2[i][2]-xmin
        h = dic2[i][3]-ymin
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
    


