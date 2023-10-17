import scipy.io as sio
import numpy as np
import pandas as pd

'''#%%
###
# 导入数据
###
file_path = r"F:\认知科学与基础\Datasets\DEAP\data_preprocessed_matlab\s01.mat"
mat = sio.loadmat(file_path)#读取mat文件
data = mat['data']  # (40, 40, 8064)
labels = mat['labels']  # (40, 4)

###
# 标签处理（二分）
###
#生成了一个标签的40长度的一维矩阵
valence_labels = []
for i in range(len(labels[:, 0])):#40
    if labels[i, 0] <= 5:#电压值小于5 为0
        valence_labels.append(0)
    else:                 #电压大于5为1
        valence_labels.append(1)
valence_labels = np.array(valence_labels)
# print(valence_labels)

arousal_labels = []
for i in range(len(labels[:, 1])):
    if labels[i, 1] <= 5:
        arousal_labels.append(0)
    else:
        arousal_labels.append(1)
arousal_labels = np.array(arousal_labels)


# print(arousal_labels)'''


###
# 特征处理
###
def calc_features(data):
    result = []
    result.append(np.mean(data))
    result.append(np.median(data))
    result.append(np.max(data))
    result.append(np.min(data))
    result.append(np.std(data))
    result.append(np.var(data))
    result.append(np.max(data) - np.min(data))
    result.append(pd.Series(data).skew())
    result.append(pd.Series(data).kurt())
    return result


'''data = data[:, :, 128 * 3:]  # (40, 40, 7680)
featured_data = np.zeros([40, 40, 101])
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range(10):
            featured_data[i, j, k * 9:(k + 1) * 9] = calc_features(data[i, j, k * 128 * 6:(k + 1) * 128 * 6])
        featured_data[i, j, 10 * 9:11 * 9] = calc_features(data[i, j, :])
        featured_data[i, j, 99] = j
        featured_data[i, j, 100] = 1


# print(featured_data.shape)
# print(valence_labels.shape)
# print(arousal_labels.shape)'''


# 处理所有被试数据
def process_labels(labels):
    # print(labels.shape)
    result = []
    for i in range(len(labels)):
        if labels[i] <= 5:
            result.append(0)
        else:
            result.append(1)
    result = np.array(result)
    # print(result.shape)
    return result


def calc_subject_featured_data(data, flag):
    data = data[:, :, 128 * 3:]  # (40, 40, 7680)#将63秒中的前三秒去掉。128*60
    featured_data = np.zeros([40, 40, 101])
    for i in range(data.shape[0]):#40
        for j in range(data.shape[1]):#40
            for k in range(10):#1-10
            #得到128*6数据的9个信息
                featured_data[i, j, k * 9:(k + 1) * 9] = calc_features(data[i, j, k * 128 * 6:(k + 1) * 128 * 6])
            featured_data[i, j, 10 * 9:11 * 9] = calc_features(data[i, j, :])
            featured_data[i, j, 99] = j
            featured_data[i, j, 100] = flag
    return featured_data


import os

file_path = r"F:\认知科学与基础\Datasets\DEAP\data_preprocessed_matlab"

data = np.zeros([32, 40, 40, 101])
valence_labels = np.zeros([32, 40])
arousal_labels = np.zeros([32, 40])

files = os.listdir(file_path)
files = files[2:]  # 去掉文件夹中的隐藏文件
for i in range(len(files)):#32
    print(file_path + '\\' + str(files[i]))
    mat = sio.loadmat(file_path + '\\' + str(files[i]), verify_compressed_data_integrity=False)#读取文件的数据
    data[i, :, :, :] = calc_subject_featured_data(mat['data'], i + 1)
    valence_labels[i, :] = process_labels(mat["labels"][:, 0])
    arousal_labels[i, :] = process_labels(mat["labels"][:, 1])
print(data[1, 1, 1, :])
print(arousal_labels[13, :])

np.save('outfile1', data)
np.save('outfile2', valence_labels)
np.save('outfile3', arousal_labels)
