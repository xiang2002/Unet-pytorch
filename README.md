# Unet-pytorch
我的输入数据尺寸：（512*512）

自然环境以及白背景下单片西瓜叶片分割模型以及利用MIOU（平均交并比）算法进行评估、分割结果展示
(由于数据集未全部上传，运行本项目代码的train.py或者accuracy.py文件皆有可能会报错，建议自己替换数据集并修改代码!)

# src目录
存放Unet模型代码及训练、预测代码，由于训练数据为我们学校私有，不便全部上传，仅上传18张原图片与标注图以便测试
    ### Unet_model.py  
        Unet模型构建代码保存在此文件内
        实例化Unet类时传入两个参数
            1：输入图片的通道数
            2：输出图片的通道数
    ### loaddata.py  
        利用 torch.utils.datar 中的 Dataset,DataLoade 来读取数据，构建数据导入器  
        由于数据集不可外传，DealDataset类不可直接运行，需要自己修改或重新创建本类  
    ### train.py  
        定义模型的训练过程，使用BCEWithLogitsLoss计算损失，用RMSprop作为模型优化器  
        如果你修改了train_loader的batchsize,请将train()中的batch_size参数一同修改  
        本模型传入的图像为(512,512,1)的灰度图片  
    ### accuracy.py  
        定义了一个MIOU（平均交并比）算法,accuracy()函数为主函数.  
        传入参数为 一张待分割的图片 , 该图片对应的标注图 , 以及待评估的模型.  
        自定义一个循环调用该accuracy函数,将结果保存在数组中,取平均值即为近似模型分割准确率.  
# model目录
存放已经训练好的一个pytorch模型文件
自测分割准确率93%！


欢迎提交Issue!  
QQ：2014176995  
Q群：733909231  
有问题欢迎联系我！  
共同学习一起进步！
