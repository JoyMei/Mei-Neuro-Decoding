import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import os
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 消除警告


# Calculating accuracy and loss
def testing(M, L, model):
    '''
    arguments:  M: testing dataset
                L: testing dataset label
                model: scikit-learn model

    return:     void
    '''
    output = model.predict(M[0:78080:32])
    label = L[0:78080:32]

    k = 0
    l = 0

    for i in range(len(label)):
        k = k + (output[i] - label[i]) * (output[i] - label[i])  # 方差

        # a good guess
        if (output[i] > 5 and label[i] > 5):
            l = l + 1
        elif (output[i] < 5 and label[i] < 5):
            l = l + 1

    print("l2 error:", k / len(label), "classification accuracy:", l / len(label), l, len(label))


# loading training and testing dataset
file_path = r"C:\Users\95933\PycharmProjects\认知科学与基础"
with open(file_path + '\data_training.npy', 'rb') as fileTrain:
    X = np.load(fileTrain)
with open(file_path + '\label_training.npy', 'rb') as fileTrainL:
    Y = np.load(fileTrainL)

X = normalize(X)  # 归一化
Z = np.ravel(Y[:, [1]])  # 扁平化

Valence_Train = np.ravel(Y[:, [0]])
Arousal_Train = np.ravel(Y[:, [1]])
Domain_Train = np.ravel(Y[:, [2]])
Like_Train = np.ravel(Y[:, [3]])

with open(file_path + '\data_validation.npy', 'rb') as fileTrain:
    M = np.load(fileTrain)

with open(file_path + '\label_validation.npy', 'rb') as fileTrainL:
    N = np.load(fileTrainL)

M = normalize(M)
L = np.ravel(N[:, [1]])  # arousa标签

Valence_Test = np.ravel(N[:, [0]])
Arousal_Test = np.ravel(N[:, [1]])
Domain_Test = np.ravel(N[:, [2]])
Like_Test = np.ravel(N[:, [3]])


def preprocess(y):
    result = []
    for i in range(len(y)):
        if y[i] <= 5:
            result.append(0)
        else:
            result.append(1)
    result = np.array(result)

    return result


# Pull out columns for X (data to train with) and Y (value to predict)拉出X（要训练的数据）和Y（要预测的值）的列
X_training = X[0:468480:32]
Y_training = Z[0:468480:32]

# Pull out columns for X (data to train with) and Y (value to predict)拉出X（要训练的数据）和Y（要预测的值）的列
X_testing = M[0:78080:32]
Y_testing = L[0:78080:32]

# DO Scale both the training inputs and outputs对训练的输入输出进行衡量
X_scaled_training = pd.DataFrame(data=X_training).values
Y_scaled_training = pd.DataFrame(data=Y_training).values

# It's very important that the training and test data are scaled with the same scaler.使用同一个定标器对训练和测试数据进行缩放是非常重要的。
X_scaled_testing = pd.DataFrame(data=X_testing).values
Y_scaled_testing = pd.DataFrame(data=Y_testing).values


# 训练集标签处理
Y_scaled_training = tf.reshape(Y_scaled_training, [-1])
Y_scaled_training = preprocess(Y_scaled_training)
Y_scaled_training = tf.cast(Y_scaled_training, dtype=tf.int32)
Y_scaled_training = tf.one_hot(Y_scaled_training, 2)

# 测试机标签处理
Y_scaled_testing = tf.reshape(Y_scaled_testing, [-1])
Y_scaled_testing = preprocess(Y_scaled_testing)
Y_scaled_testing = tf.cast(Y_scaled_testing, dtype=tf.int32)
Y_scaled_testing = tf.one_hot(Y_scaled_testing, 2)

# print(tf.shape(X_scaled_training), np.array(X_scaled_training).shape)
# print(tf.shape(Y_scaled_training), np.array(Y_scaled_training).shape)
# print(Y_scaled_training[:100])


model = tf.keras.Sequential([
    tf.keras.layers.Dense(70),  # 输入层
    tf.keras.layers.Dense(512, activation=tf.nn.relu),  # 第一层
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),  # 第二层
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),  # 第三层
    tf.keras.layers.Dense(512, activation=tf.nn.relu),  # 第四层
    # 输出层，2分类问题，激活函数为softmax
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)  # 输出层
])

# 输出一下模型的结构
model.build((None, 70))
model.summary()
# 编译模型，配置优化器、损失函数以及监控指标
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
# 调用tf.keras封装的训练接口，开始训练
history = model.fit(x=X_scaled_training, y=Y_scaled_training, epochs=500, validation_data=(X_scaled_testing, Y_scaled_testing))
# 画图
history.history.keys()
plt.plot(history.epoch, history.history.get('accuracy'), label='accuracy')
plt.plot(history.epoch, history.history.get('val_accuracy'), label='val_accuracy')
plt.legend()
plt.show()
