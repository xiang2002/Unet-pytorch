import torch
from torch.utils.data import Dataset,DataLoader
from PIL import Image


#自定义Dataset的一个子类
class DealDataset(Dataset):

    def __init__(self):
        data=[]
        #读取图片数据的存储路径
        for i in range(1,3933):
            data.append(("data/input/"+str(i).zfill(4)+"picard.png","data/output/"+str(i).zfill(4)+"labelard.png"))
        for i in range(1,3933):
            data.append(("data/LR/"+str(i).zfill(4)+"LR.png","data/LRL/"+str(i).zfill(4)+"LRL.png"))
        for i in range(1,3933):
            data.append(("data/TB/"+str(i).zfill(4)+"TB.png","data/TBL/"+str(i).zfill(4)+"TBL.png"))
        self.images=[]
        self.labels=[]
        for i,(img,lbl) in enumerate(data):
            # print(i)
            #尝试读取数据并做归一化处理
            try:
                image = torch.from_numpy(np.array(Image.open(img).convert('L'))/255)
                label = torch.from_numpy(np.array(Image.open(lbl))/255)
                self.images.append(image)
                self.labels.append(label)
            except Exception as e:
                continue
    
    def __getitem__(self, index):

        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)


dtset = DealDataset()
#设置batch为8，一批训练八张图片，可随机器配置修改数值
train_loader = DataLoader(dataset=dtset,
                          batch_size=8)