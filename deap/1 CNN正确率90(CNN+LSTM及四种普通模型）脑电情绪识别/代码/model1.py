import tensorflow as tf
from tensorflow.python.keras.layers import Dropout
from data import load_data_1d

tf.compat.v1.disable_eager_execution()
###
# tf.keras构建全连接模型
###

# 加载数据集
train_db, dev_db = load_data_1d(128)

# 构建模型
model = tf.keras.Sequential([
    # 第一层，节点为512，激活函数为relu
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    # 输出层，2分类问题，激活函数为softmax
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

# 输出一下模型的结构
model.build((None, 4040))
model.summary()
# 编译模型，配置优化器、损失函数以及监控指标
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])
# 调用tf.keras封装的训练接口，开始训练
model.fit_generator(train_db, epochs=500, validation_data=dev_db)
