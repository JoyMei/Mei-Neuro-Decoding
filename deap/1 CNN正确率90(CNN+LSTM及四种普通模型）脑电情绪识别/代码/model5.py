from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras import regularizers
from keras.optimizers import Adam
from data import load_data_2d
import tensorflow as tf

train_db, dev_db = load_data_2d(128)


# RNN网络结构

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(512, input_shape=(40, 101), activation='relu', return_sequences=True, dropout=0.2),
    tf.keras.layers.LSTM(512, activation='relu', dropout=0.2),
    tf.keras.layers.Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0.01))
])


# 输出网络结构
model.build((None, 40, 101))
model.summary()

# 编译模型
opt = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 训练模型
model.fit(train_db, epochs=500, validation_data=dev_db)
