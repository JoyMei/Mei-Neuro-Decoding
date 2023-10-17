from random import seed
import tensorflow as tf
from data import load_data_1d

tf.compat.v1.disable_eager_execution()
###
# tf.keras构建深度神经网络（DNN结构）
###

# 加载数据集
train_db, dev_db = load_data_1d(128)

# 构建模型
init = tf.keras.initializers.glorot_uniform(seed == 1)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=5, kernel_initializer=init, activation='relu'),
    tf.keras.layers.Dense(units=6, kernel_initializer=init, activation='relu'),
    tf.keras.layers.Dense(units=2, kernel_initializer=init, activation='softmax')
])

# 输出一下模型的结构
model.build((None, 4040))
model.summary()
# 编译模型，配置优化器、损失函数以及监控指标
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])
# 调用tf.keras封装的训练接口，开始训练
model.fit_generator(train_db, epochs=1000, validation_data=dev_db)
