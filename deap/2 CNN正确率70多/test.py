import scipy.io as sio
import numpy as np
import pandas as pd
import os
import torch     
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import math
import h5py
import os
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
# class CNN_DEAP(nn.Module):
#     def __init__(self, num_class, input_size):
#         super(CNN_DEAP, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(1, 100 ,kernel_size=3),
#             nn.Tanh(),
#             nn.Conv2d(100, 100, kernel_size=3), 
#             # nn.Tanh(),
#             # nn.Conv2d(200,200,kernel_size=3),
#             nn.MaxPool2d(kernel_size=2, ),
#             nn.Dropout(p=0.5)
#         )
#         self.BN_s = nn.BatchNorm2d(100)
#         self.size = self.getsize(input_size)
# #         print(self.size[1])96000
# #         print(self.size[0])1
#         self.classifier = nn.Sequential(
#             #构建pytorch中神经网络的(nn)的线性层，且输入特征为self.size[1]个，输出特征为48
#             nn.Linear(self.size[1], 128),
#             nn.Tanh(),
#             nn.Dropout(p=0.35),
#             nn.Linear(128, 48),
#             nn.Tanh(),
#             nn.Dropout(p=0.25),
#             nn.Linear(48,num_class)
#         )

#     def forward(self, x):#前向传播
# #         print(x.shape)
#         x = self.features(x)
#         # print("size:"+str(self.size))
# #         print(x.shape)    
#         x = x.view(x.size()[0],-1)
#         x = self.classifier(x)
#         return x

#     def getsize(self, input_size):
#         data = torch.ones(1, 1, input_size[0], input_size[1])
# #         print(data)
#         x = self.features(data)
# #         print(x)
#         out = x.view(x.shape[0], -1)
# #         print(out)tensor([[0.0000, 0.3013, 0.3013,  ..., 0.0000, 0.0000, 0.0650]],
# #        grad_fn=<ViewBackward>)
#         return out.size()


# model = CNN_DEAP(2, [40, 101])
# print(model)
# print("=====================")
class CNN_DEAP(nn.Module):
    def __init__(self, num_class, input_size):
        super(CNN_DEAP, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=16,  kernel_size=(1,64), stride=1, padding=0),
        #     nn.Tanh(),
        #     nn.MaxPool2d(kernel_size=(1,)),
        #     nn.Dropout(p=0.25),
        #     # nn.BatchNorm2d(num_features=16, eps=1e-04, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(int(input_size[0]*0.5), 1), stride=1, padding=0),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.25),
        #     nn.MaxPool2d(kernel_size=(1,)),
        #     # nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(16, 1), stride=1, padding=0)
        # )

        # self.features = nn.Sequential(
        #     nn.Conv2d(in_channels=40, out_channels=20,  kernel_size=(5,5), stride=(1,5), padding=0),
        #     nn.Tanh(),
        #     nn.MaxPool2d(kernel_size=(2,2)),
        #     nn.Dropout(p=0.25),
        #     # nn.BatchNorm2d(num_features=16, eps=1e-04, momentum=0.1, affine=True, track_running_stats=True),
        #     nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(3,3), stride=(1,2), padding=0),
        #     nn.Tanh(),
        #     nn.Dropout(p=0.25),
        #     nn.MaxPool2d(kernel_size=(2,2),stride=(1,1))
        #     # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=( 1,64), stride=(1,2), padding=0),
        #     # nn.MaxPool2d(kernel_size=(2,2)),
        # )

        self.features = nn.Sequential(
            #构建pytorch中神经网络的(nn)的线性层，且输入通道1个，输出通道16，卷积核40*1 步长为1
            nn.Conv2d(in_channels=1, out_channels=16,  kernel_size=(int(input_size[0]),1), stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(1,)),
            nn.Dropout(p=0.3),
        )
          
        self.features2 =nn.Sequential(
             #构建pytorch中神经网络的(nn)的线性层，且输入通道1个，输出通道16，卷积核20*1 步长为1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(int(input_size[0]*0.5), 1), stride=1, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(1,)),
            nn.Dropout(p=0.3)
          
        )
        self.size = self.getsize(input_size)
#      

        self.classifier = nn.Sequential(
            #构建pytorch中神经网络的(nn)的线性层，且输入特征为self.size[1]个，输出特征为128
            nn.Linear(self.size[1],128),
            nn.Tanh(),
            nn.Dropout(p=0.35),
            #构建pytorch中神经网络的(nn)的线性层，且输入特征为128个，输出特征为48
            nn.Linear(128,48),
            nn.Tanh(),
            nn.Dropout(p=0.3),
            #构建pytorch中神经网络的(nn)的线性层，且输入特征为48个，输出特征为2
            nn.Linear(48,num_class)
            )

    def forward(self, x):#前向传播
#         print(x.shape)
        x1 = self.features(x)
        x2 = self.features2(x)
#         print(x)
        x = torch.cat((x1,x2),dim=2)
 
        x = x.view(x1.size()[0],-1)
        x = self.classifier(x)

        return x

    def getsize(self, input_size):
        data = torch.ones((1, 1, input_size[0], input_size[1]))
        x1 = self.features(data)

        x2 = self.features2(data)
        #将输出的特征整合为一个二维特征
        x = torch.cat((x1,x2),dim=2)
        out = x.view(x.shape[0], -1)
# 
        return out.size()


model = CNN_DEAP(2, [40, 101])
print(model)
print("=====================")

class EEGDataset(Dataset):

    def __init__(self, x_tensor, y_tensor):

        self.x = x_tensor
        self.y = y_tensor

        assert self.x.size(0) == self.y.size(0)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.y)
    
class TrainModel():
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.regulization = regulization(model, self.Lambda)
    
    def load_data(self, path,subject):
        path = Path(path)
        data = np.zeros([32,40,40,101])
        labels = np.zeros([32,40])
        file_code = 'sub_31.hdf'
        file = path / file_code
        data_dictionary = h5py.File(file, 'r')
        for i in range(subject):
            data[i,:,:,:] = np.array(data_dictionary['data'][i])
            labels[i,:] =np.array(data_dictionary['valence_labels'][i])
            # labels[i,:] = np.array(data_dictionary['arousal_labels'][i])
        print('The shape of data is:'+ str(data[0,0].shape))
        print('The shape of label is:' + str(labels[0].shape))
        return data,labels
    
    def set_parameter(self, num_classes,data,labels,input_size,learning_rate,batch_size,epoch,patient):
        self.num_classes = num_classes
        self.data = data
        self.labels = labels
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.batch_size = batch_size
        self.patient = patient
        self.epoch = epoch    

    def leave_one_subject_out(self):#1
        subjects = self.data.shape[0]
#         print(self.data.shape[0])#32

        ACC = []
        mean_ACC = []
        LOSS_TEST = []
        ACC_TEST = []
        for i in range(subjects-1):
            print("====begin=====")
            index = np.arange(subjects - 1)
            index_train = np.delete(index, i)
            # index_train = np.delete(index, i)
            index_test = i
            # print(index)
            # print(index[i:i+2])

            # print(self.data.shape)#(32, 40, 40, 101)
            # print(self.labels.shape)#(32, 40)

            # 划分数据集
            data_test1 = self.data[index_test, :, :, :]   
            # print(type(data_test1))
            data_test3 = self.data[subjects - 1,:,:,:]
            # data_test2 = self.data[i+2,:,:,:]
            data_test = np.vstack((data_test1,data_test3))
            # data_test = self.data[index_test, :, :, :]

            labels_test1 = self.labels[index_test, :]
            labels_test3 = self.labels[subjects - 1,:]
            # labels_test2 = self.labels[i+2,:]
            labels_test = np.hstack((labels_test1,labels_test3))
            # labels_test = self.labels[index_test, :]
            print("Test:", data_test.shape, labels_test.shape)

            data_train = self.data[index_train, :, :, :]
#             print(data_train.shape)(31, 40, 40, 101)
            labels_train = self.labels[index_train, :]

            data_train, labels_train, data_val, labels_val = self.split(data_train, labels_train)
        
            # 增加深度维度
            data_train = data_train[:,np.newaxis,:,:]
#             print(data_train.shape)(992, 1, 40, 101)
            data_val = data_val[:,np.newaxis,:,:]
            data_test = data_test[:,np.newaxis,:,:] 

            # 转换数据格式
            data_train = torch.from_numpy(data_train).float()
            labels_train = torch.from_numpy(labels_train).long()
            data_val = torch.from_numpy(data_val).float()
            labels_val = torch.from_numpy(labels_val).long()

#             print(data_test.shape)#(40, 1, 40, 101)
#             print(labels_test.shape)#(40,)

            data_test = torch.from_numpy(data_test).float()
            labels_test = torch.from_numpy(labels_test).long()

            print("Training:", data_train.size(), labels_train.size())
            #Training: torch.Size([992, 1, 40, 101]) torch.Size([992])
            
            print("Validation:", data_val.size(), labels_val.size())#Validation: torch.Size([248, 1, 40, 101]) torch.Size([248])
            print("Test:", data_test.size(), labels_test.size())#Test: torch.Size([40, 1, 40, 101]) torch.Size([40])
            

            ACC_one_sub = self.train(data_train, labels_train,data_test, labels_test,data_val,labels_val)

            ACC.append(ACC_one_sub)
            print("Subject:" + str(i) +"\nAccuracy:%.2f" % ACC_one_sub)
            ACC_TEST.append(ACC_one_sub)
            file = open("result_session_PCNN.txt","a")
            file.write(str(model)+'\n')
            file.write('Subject:'+str(i)+'\nAccuracy:'+str(ACC_one_sub)+'\n')
            file.close()
        
        ACC = np.array(ACC)
        mean_ACC = np.mean(ACC)
        std = np.std(ACC)
        print(ACC)
        print("std: %.2f" % std)
        print("*"*20)
        print("Mean accuracy of model is: %.2f" % mean_ACC)
        #log the results per subject
        file = open("result_session_PCNN.txt","a")
        file.write('Subject:'+str(i)+' MeanACC:'+str(mean_ACC)+' Std:'+str(std)+'\n')
        file.close()


    def split(self, data, label):
        np.random.seed(0)
        data = np.concatenate(data, axis=0)
#         print(data.shape)(1240, 40, 101)
        label = np.concatenate(label, axis=0)##对于一维数组拼接，axis的值不影响最后的结果
#         print(label.shape)1240
    
        
        index = np.arange(data.shape[0])
#         print(index)shape[   0    1    2 ... 1237 1238 1239]
#         print(index.shape)(1240,)
        index_random = index
#         print(index_random)[   0    1    2 ... 1237 1238 1239]
        
        np.random.shuffle(index_random)#np.random,shuffle作用就是重新排序返回一个随机序列
#         print(index_random)[1030  124  184 ... 1216  559  684]
        datas= data[index_random]
#         print(data.shape)    (1240, 40, 101)
        labels = label[index_random]
#         print(labels.shape)(1240,)
         
        # get validation set     print("os")
        val_data = datas[int(data.shape[0]*0.8):]
#         print(val_data.shape)(248, 40, 101)
        val_labels = labels[int(data.shape[0]*0.8):]
#         print(val_labels.shape)(248,)

        # get train set
        train_data =data[0:int(data.shape[0]*0.8)]
#         print(train_data.shape)(992, 40, 101)
        train_labels = label[0:int(data.shape[0]*0.8)]
#         print(train_data.shape)(992, 40, 101)
     
        return train_data, train_labels, val_data, val_labels
                                                                        
    def make_train_step(self, model, loss_fn, optimzier):
        def train_step(x, y):
            model.train()
            # torch.Size([50, 1, 40, 101])
            yhat = model(x)
            pred = yhat.max(1)[1]
            correct = (pred == y).sum()
            acc = correct.item()/len(pred)
            # loss_r = self.regulization(model, self.Lambda)
            loss = loss_fn(yhat, y)
            # loss = loss_fn(yhat)
            optimzier.zero_grad()#清空过往梯度；
            loss.backward(retain_graph=True) #反向传播，计算当前梯度；
            loss.backward()# 反向传播，计算当前梯度；
            optimzier.step() #根据梯度更新网络参数
            # optimzier.step()
            return loss.item(), acc
        return train_step

    def train(self, train_data, train_label, test_data, test_label, val_data,val_label):
        print("Available Device:" + self.device)#Available Device:cpu
        
        model = CNN_DEAP(self.num_classes, self.input_size)
        
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()#交叉熵损失函数
        # loss_fn = nn.Softmax()
        if torch.cuda.is_available():
            model = model.to(self.device)
            loss_fn = loss_fn.to(self.device)

        train_step = self.make_train_step(model, loss_fn, optimizer)


        # 导入数据
        dataset_train = EEGDataset(train_data, train_label)

        dataset_val = EEGDataset(val_data, val_label)

        dataset_test = EEGDataset(test_data, test_label)


        # 创建DataLoader
        train_loader = DataLoader(dataset=dataset_train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=dataset_val, batch_size=self.batch_size,shuffle=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=self.batch_size, shuffle=True)
        
        losses = []
        accs = []
        Acc_val = []
        Loss_val = []
        val_losses = []
        val_acc = []

        test_losses = []
        test_acc = []
        Acc_test = []

        Acc = []
        acc_max = 0
        patient = 0
        file = open("result_session_PCNN.txt","a")
        file.write('\nTIME_BEGIN:'+str(datetime.datetime.now())+'\n')

        ep = 0
        loss1 = []
        acc1= []
        Eloss1 = []
        Eacc1 = []
        for epoch in range(self.epoch):
            loss_epoch = []
            acc_epoch = []
            for i, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(self.device)#to强制类型转换
                y_batch = y_batch.to(self.device)                  
                loss, acc = train_step(x_batch, y_batch)
                loss_epoch.append(loss)
                acc_epoch.append(acc)
                              
            losses.append(sum(loss_epoch)/len(loss_epoch))
            accs.append(sum(acc_epoch)/len(acc_epoch))
            loss_epoch = []
            acc_epoch = []
            print('Epoch [{}/{}], Loss:{:.4f}, Acc:{:.4f}'.format(epoch+1, self.epoch, losses[-1], accs[-1]))
            acc1.append( accs[-1])
            loss1.append(losses[-1])
            ep += 1

            file = open("result_session_PCNN.txt","a")
            file.write('Epoch [{'+str(epoch+1)+'}/{'+str(self.epoch)+'}], Loss:'+str(losses[-1])+', Acc:'+str(accs[-1])+'\n')
            file.close()
            ##############Validation process####################
            with torch.no_grad():                                                                          
                for x_val, y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val = y_val.to(self.device)

                    model.eval()

                    yhat = model(x_val)
                    pred = yhat.max(1)[1]
                    correct = (pred == y_val).sum()
                    acc = correct.item()/len(pred)
                    val_loss = loss_fn(yhat, y_val)
                    val_losses.append(val_loss.item())
                    val_acc.append(acc)

                Acc_val.append(sum(val_acc)/len(val_acc))
                Loss_val.append(sum(val_losses)/len(val_losses))
                print('Evaluation Loss:{:.4f}, Acc:{:.4f}'.format(Loss_val[-1], Acc_val[-1]))
                Eloss1.append(Loss_val[-1])
                Eacc1.append(Acc_val[-1])
                file = open("result_session_PCNN.txt","a")
                file.write('Evaluation Loss:'+str(Loss_val[-1])+', Acc:'+str(Acc_val[-1])+'\n')
                file.close()
                val_losses = []
                val_acc = []

                ######## early stop ########
            Acc_es = Acc_val[-1]

            if Acc_es > acc_max:
                acc_max = Acc_es
                patient = 0
                torch.save(model,'valence_max_model.pt')
                print('----Model saved!----')
            else:
                patient += 1
            if patient > self.patient:
                print('-----Early stoping----')
                break
    
        acc1 = np.array(acc1)
        Eacc1 = np.array(Eacc1)
        print(acc1)
        print(Eacc1)
        loss1 = np.array(loss1)
        Eloss1 = np.array(Eloss1)
        print(loss1)
        print(Eloss1)

        #########test process############
        model = torch.load('valence_max_model.pt')
        with torch.no_grad():                                                       
            for x_test, y_test in test_loader:
                x_test = x_test.to(self.device)
                y_test = y_test.to(self.device)
                model.eval()
                yhat = model(x_test)
                pred = yhat.max(1)[1]
                correct = (pred == y_test).sum()
                acc = correct.item()/len(pred)
                test_loss = loss_fn(yhat, y_test)
                test_losses.append(test_loss.item())
                test_acc.append(acc)
                                                                                              
            print('Test Loss:{:.4f}, Acc:{:.4f}'
                  .format(sum(test_losses)/len(test_losses), sum(test_acc)/len(test_acc)))
            file = open("result_session_PCNN.txt","a")
            file.write('Test Loss:'+str(sum(test_losses)/len(test_losses))+', Acc:'+str(sum(test_acc)/len(test_acc))+'\n')
            file.close()
            Acc_test = (sum(test_acc)/len(test_acc))

        return Acc_test    


ACC_SUBJECT = []
data = np.zeros([32,40,40,101])
labels = np.zeros([32,40])
train = TrainModel()
data,labels = train.load_data(path = 'D:\pythonProject1\DEAP\CNN\Datasets',subject = 32)
# print(data[0,0].shape)
train.set_parameter(num_classes=2,
                data=data,
                labels=labels,
                input_size=data[0,0].shape,
                learning_rate=0.0001,
                batch_size=68,
                epoch=60,
                patient= 4,
                )
# print(data[0,0].shape)
ACC_SUBJECT = train.leave_one_subject_out()

subject = []
for i in range(31):
    subject.append(i)
plt.plot(subject,ACC_SUBJECT,color='Red')
plt.title('accuracy_changing')
plt.xlabel('subject')
plt.ylabel('acc')
plt.savefig('./cnn.jpg')
plt.show()