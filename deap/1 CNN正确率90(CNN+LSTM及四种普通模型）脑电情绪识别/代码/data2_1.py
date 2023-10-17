import numpy as np
import pyeeg as pe
import pickle as pickle
import pandas as pd
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

import os
import tensorflow as tf
import time

# 定义全局变量
channel = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31]  # 14 Channels chosen to fit Emotiv Epoch+
band = [4, 8, 12, 16, 25, 45]  # 5 bands
window_size = 256  # Averaging band power of 2 sec平均频带功率为2秒
step_size = 16  # Each 0.125 sec update once每0.125秒更新一次
sample_rate = 128  # Sampling rate of 128 Hz采样率128 Hz
subjectList = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
               '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32']


# FFT with pyeeg
def FFT_Processing(sub, channel, band, window_size, step_size, sample_rate):
    meta = []
    file_path = r"F:\认知科学与基础\Datasets\DEAP\data_preprocessed_python"
    with open(file_path + '\s' + sub + '.dat', 'rb') as file:

        subject = pickle.load(file, encoding='latin1')  # resolve the python 2 data problem by encoding : latin1

        for i in range(0, 40):
            # loop over 0-39 trails
            data = subject["data"][i]
            labels = subject["labels"][i]
            start = 0

            while start + window_size < data.shape[1]:
                meta_array = []
                meta_data = []  # meta vector for analysis
                for j in channel:
                    X = data[j][start: start + window_size]  # Slice raw data over 2 sec, at interval of 0.125 sec以0.125秒为间隔，在2秒内对原始数据进行切片
                    Y = pe.bin_power(X, band,
                                     sample_rate)  # FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
                    meta_data = meta_data + list(Y[0])

                meta_array.append(np.array(meta_data))
                meta_array.append(labels)

                meta.append(np.array(meta_array))
                start = start + step_size

        meta = np.array(meta)
        np.save('out' + sub, meta, allow_pickle=True, fix_imports=True)


for subjects in subjectList:
    FFT_Processing(subjects, channel, band, window_size, step_size, sample_rate)

# 数据划分
# training dataset: 75 %
# validation dataset: 12.5%
# testing dataset: 12.5%

data_training = []
label_training = []
data_testing = []
label_testing = []
data_validation = []
label_validation = []

for subjects in subjectList:
    file_path = r"C:\Users\95933\PycharmProjects\认知科学与基础"
    with open(file_path + '\out' + subjects + '.npy', 'rb') as file:
        sub = np.load(file, allow_pickle=True)
        for i in range(0, sub.shape[0]):
            if i % 8 == 0:
                data_testing.append(sub[i][0])
                label_testing.append(sub[i][1])
            elif i % 8 == 1:
                data_validation.append(sub[i][0])
                label_validation.append(sub[i][1])
            else:
                data_training.append(sub[i][0])
                label_training.append(sub[i][1])

np.save('data_training', np.array(data_training), allow_pickle=True, fix_imports=True)
np.save('label_training', np.array(label_training), allow_pickle=True, fix_imports=True)
print("training dataset:", np.array(data_training).shape, np.array(label_training).shape)

np.save('data_testing', np.array(data_testing), allow_pickle=True, fix_imports=True)
np.save('label_testing', np.array(label_testing), allow_pickle=True, fix_imports=True)
print("testing dataset:", np.array(data_testing).shape, np.array(label_testing).shape)

np.save('data_validation', np.array(data_validation), allow_pickle=True, fix_imports=True)
np.save('label_validation', np.array(label_validation), allow_pickle=True, fix_imports=True)
print("validation dataset:", np.array(data_validation).shape, np.array(label_validation).shape)
