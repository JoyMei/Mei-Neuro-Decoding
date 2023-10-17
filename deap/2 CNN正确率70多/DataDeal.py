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
# #导入SEED数据
# data_path  = r'D:\pythonProject1\DEAP\Datasets\SEED\SEED_PreprocessedEEG\1_20131107.mat'
# seeds01mat = sio.loadmat(data_path)

# labels_path = r'D:\pythonProject1\DEAP\Datasets\SEED\SEED_ExtractedFeatures\label.mat'
# seeds01labels = sio.loadmat(labels_path)['label']

# seeds01labels = np.array(list(seeds01labels))
# print(seeds01labels.shape)
# np.squeeze(seeds01labels)#作用：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
# seeds01labels = seeds01labels.reshape((15))
# print(seeds01labels.shape)
# print(seeds01labels)

# #导入数据
# file_path  = r'D:\pythonProject1\DEAP\Datasets\DEAP\data_preprocessed_matlab\s01.mat'
# mat = sio.loadmat(file_path)
# # print(mat)
# data = mat['data']#(40,40,8064)
# print(data.shape)
# labels = mat['labels']#(40,4)
# print(labels.shape)

# valence_labels = []
# for i in range(len(labels[:,0])):
#     if(labels[i,0] <= 5):
#         valence_labels.append(0)
#     else:
#         valence_labels.append(1)
# valence_labels = np.array(valence_labels)
# valence_labels

# arousal_labels = []
# for i in range(len(labels[:,0])):
#     if(labels[i,0] <= 5):
#         arousal_labels.append(0)
#     else:
#         arousal_labels.append(1)
# arousal_labels = np.array(arousal_labels)
# arousal_labels

def calc_features(data):
    result = []
    result.append(np.mean(data))#平均值
    result.append(np.median(data))#中值
    result.append(np.max(data))#最大值
    result.append(np.min(data))#最小值
    result.append(np.std(data))#标准差
    result.append(np.var(data))#方差
    result.append(np.max(data)-np.min(data))#范围
    result.append(pd.Series(data).skew())#偏度
    result.append(pd.Series(data).kurt())#峰度值
    return result

#处理所有被试的数据
def process_labels(labels):
    result = []
    for i in range(len(labels)):
        if(labels[i] < 5):
            result.append(0)
        else:
            result.append(1)
    result = np.array(result)
    return result

# def calc_subject_featured_data(data, flag):
#     data = data[:, :,:] #(40, 40, 7680)
# #     print(data.shape)
#     featured_data = np.zeros([40,40,101])
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             for k in range(10):
#                 featured_data[i,j,k*9:(k+1)*9] = calc_features(data[i,j,k*128*6:(k+1)*128*6])
#             featured_data[i,j,10*9:11*9] = calc_features(data[i,j,:])
#             featured_data[i,j,99] = j
#             featured_data[i,j,100] = flag
#     return featured_data

def calc_subject_featured_data(data, flag):
    data = data[:, :,:] #(40, 40, 7680)
#     print(data.shape)
    featured_data = np.zeros([40,40,992])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(10):
                featured_data[i,j,k*90:(k+1)*90] = calc_features(data[i,j,k*128*6:(k+1)*128*6])
            featured_data[i,j,900:990] = calc_features(data[i,j,:])
            featured_data[i,j,990] = j
            featured_data[i,j,991] = flag
    return featured_data

file_path = r"D:\pythonProject1\DEAP\Datasets\DEAP\data_preprocessed_matlab"
data = np.zeros([32,40,40,992])
valence_labels = np.zeros([32,40])
arousal_labels = np.zeros([32,40])
files = os.listdir(file_path)
files = files[2:] # 去掉文件夹中的隐藏文件
for i in range(len(files)):
# print(file_path+'\\'+str(files[i]))
    mat = sio.loadmat(file_path+'\\'+str(files[i]))
    data[i,:,:,:] = calc_subject_featured_data(mat['data'], i+1)
    valence_labels[i,:] = process_labels(mat["labels"][:,0])
    arousal_labels[i,:] = process_labels(mat["labels"][:,1])
    print(i)
    #Log the results per subject
filename_data =  Path('D:\pythonProject1\DEAP\CNN\Datasets\sub_33.hdf')
save_data  = h5py.File(filename_data,'w')
save_data['data'] = data
save_data['valence_labels'] = valence_labels
save_data['arousal_labels'] = arousal_labels
save_data.close()
print("====isDealed====")
print(data[1,1,1,:].shape)
print(arousal_labels[13,:].shape)