import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import tensorflow as tf

# loading training and testing dataset
from tensorflow.python.keras.models import Sequential

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


# Pull out columns for X (data to train with) and Y (value to predict)
X_training = X[0:468480:32]
Y_training = Z[0:468480:32]

# Pull out columns for X (data to train with) and Y (value to predict)
X_testing = M[0:78080:32]
Y_testing = L[0:78080:32]

# DO Scale both the training inputs and outputs
X_scaled_training = pd.DataFrame(data=X_training).values
Y_scaled_training = pd.DataFrame(data=Y_training).values

# It's very important that the training and test data are scaled with the same scaler.
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

X_scaled_training = tf.reshape(X_scaled_training, (-1, 70, 1))
X_scaled_testing = tf.reshape(X_scaled_testing, (-1, 70, 1))

###
# tf.keras构建1D 卷积和 LSTM 混合模型
###


model = Sequential()
model.add(tf.keras.layers.Conv1D(64, 15, strides=2, input_shape=(70, 1), use_bias=False))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.Conv1D(64, 3))
model.add(tf.keras.layers.Conv1D(64, 3, strides=2))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.LSTM(512, dropout=0.5, return_sequences=True))
model.add(tf.keras.layers.LSTM(256, dropout=0.5, return_sequences=True))
model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dense(2, activation="softmax"))

# 输出一下模型的结构
model.build((None, 70, 1))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
# 调用tf.keras封装的训练接口，开始训练
history = model.fit(x=X_scaled_training, y=Y_scaled_training, epochs=1000,
                    validation_data=(X_scaled_testing, Y_scaled_testing))

# 画图
history.history.keys()
plt.plot(history.epoch, history.history.get('accuracy'), label='accuracy')
plt.plot(history.epoch, history.history.get('val_accuracy'), label='val_accuracy')
plt.legend()
plt.show()

# 模型保存
model.save('model2_2.h5')
