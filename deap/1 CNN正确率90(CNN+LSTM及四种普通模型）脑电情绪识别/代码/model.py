import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import np_utils

###
# 创建模型
###


data = np.load('outfile1.npy')
valence_labels = np.load('outfile2.npy')
arousal_labels = np.load('outfile3.npy')
# print(data.ndim, data.shape)
# print(valence_labels.ndim, valence_labels.shape)
# print(arousal_labels.ndim, arousal_labels.shape)

data = tf.reshape(data, [1280, -1])  # (32,40,40,101)->(1280,4040)
# print(data.shape)


valence_labels = tf.reshape(valence_labels, [1280])
nb_classes = 2
valence_labels = tf.cast(valence_labels, dtype=tf.int32)#类型变换
arousal_labels = tf.cast(arousal_labels, dtype=tf.int32)#类型变换
valence_labels = np_utils.to_categorical(valence_labels, nb_classes)  # One-hot encoding
arousal_labels = np_utils.to_categorical(arousal_labels, nb_classes)   #one-hot编码
print(valence_labels.shape)
print(arousal_labels.shape)

model = Sequential()

model.add(Dense(512, activation='relu', input_shape=(4040,), kernel_initializer='he_normal'))
# model.add(Dropout(0.2))

model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
# model.add(Dropout(0.2))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(data, valence_labels, epochs=500, batch_size=64, verbose=1, validation_split=0.05)
