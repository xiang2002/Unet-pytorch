from torch import optim
from torch.utils.data import Dataset, DataLoader
from Unet_model import Unet




#超参数自己可调
def train_net(net, device, epochs=40, batch_size=8, lr=0.00003,train_loader=train_loader):
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-8, weight_decay=0)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for i,(image, label) in enumerate(train_loader):
            # 将数据拷贝到device中
            image_data = image.reshape(batch_size,1,512,512).to(device=device, dtype=torch.float32)
            label_data = label.reshape(batch_size,1,512,512).to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image_data)
            # 计算loss
            loss = criterion(pred, label_data)
            #writer.add_scalar("loss", loss.item(), global_step=None, walltime=None)
            print("Epoch:",epoch,'Loss/train:', loss.item(),"num:",i)
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



model = Unet(1,1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_net(model.cuda(), device)