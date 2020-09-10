import numpy as np
import torch
import cv2

def generate_matrix(gt_image, pre_image,num_class=38):
        mask = (gt_image >= 0) & (gt_image < num_class)#ground truth中所有正确(值在[0, class_num])的像素label的mask
        label = num_class * gt_image[mask].astype('int') + pre_image[mask].astype("int")
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=num_class**2)
        confusion_matrix = count.reshape(num_class, num_class)#21 * 21(for pascal)
        return confusion_matrix


    
def Frequency_Weighted_Intersection_over_Union(confusion_matrix):
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))

    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU
 
    
def accuracy(n):
    xy = net(torch.from_numpy(cv2.cvtColor(test_img[n],cv2.COLOR_BGR2GRAY).reshape(512,512)).reshape(1,1,512,512).to(device=device, dtype=torch.float32)).cpu().detach().numpy().reshape(512,512).astype("int64")
    xy = (xy//255)
    xy[xy<20]=0
    xy[xy>=20]=38
    matrix =generate_matrix(xy,test_lbl[n])
    acc = Frequency_Weighted_Intersection_over_Union(matrix)
    print(acc)
    return acc
















net = Unet(n_channels=1, n_classes=1)
net.to(device=device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load("epoch9.pth", map_location=device).items()})
net.eval()