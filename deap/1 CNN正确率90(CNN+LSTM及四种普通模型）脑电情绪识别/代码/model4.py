from data import load_data_2d
import tensorflow as tf
import matplotlib.pyplot as plt


train_db, dev_db = load_data_2d(128)

###
# tf.keras构建卷积神经网络（LSTM结构）
###

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=10, input_shape=(40, 101)),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

# 输出网络结构
model.build((None, 40, 101))
model.summary()
model.compile(tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
# 开始训练
history = model.fit_generator(train_db, epochs=10, validation_data=dev_db)
# 画图
history.history.keys()
plt.plot(history.epoch, history.history.get('accuracy'), label='accuracy')
plt.plot(history.epoch, history.history.get('val_accuracy'), label='val_accuracy')
plt.legend()
plt.show()

