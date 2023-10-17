from random import seed
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)#用于生成随机数，种子相同，生成随机数相同


###
# 数据预处理
###

def preprocess_1d(x, y):
    '''
    数据预处理
    (40,40,101)->(-1,4040)
    '''
    x = tf.cast(x, dtype=tf.float32)#转换数据类型成float32 执行 tensorflow 中张量数据类型转换，比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32。
    x = tf.reshape(x, [-1, 4040])#-1=40*40*101/4040=40
    # 将标签转成one-hot形式
    y = tf.reshape(y, [-1])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, 2)#tf.one_hot()函数是将input转化为one-hot类型数据输出，相当于将多个数值联合放在一起作为多个相同类型的向量，
    #可用于表示各自的概率分布，通常用于分类任务中作为最后的FC层的输出，有时翻译成“独热”编码。
    return x, y


def load_data_1d(batch_size=128):
    '''
    加载数据集,返回一个tf.data.Dataset对象
    '''
    data = np.load('outfile1.npy')
    valence_labels = np.load('outfile2.npy')

    # arousal_labels = np.load('outfile3.npy')

    # 划分测试集与训练集（测试机占0.2）
    data_train, data_test, valence_labels_train, valence_labels_test = train_test_split(data, valence_labels,
                                                                                        test_size=0.2,
                                                                                        random_state=seed)

    # 封装成tf.data.Dataset数据集对象
    train_db = tf.data.Dataset.from_tensor_slices((data_train, valence_labels_train))#按第一维度切分
    # 设置每次迭代的mini_batch大小
    train_db = train_db.batch(batch_size)#从数组中每次获取一个batch_size的数据
    # 对数据进行预处理
    train_db = train_db.map(preprocess_1d)
    # 打乱数据顺序
    train_db = train_db.shuffle(10000)

    # 封装数据集对象
    test_db = tf.data.Dataset.from_tensor_slices((data_test, valence_labels_test))
    # 设置mini_batch
    test_db = test_db.batch(batch_size)
    # 进行数据预处理
    test_db = test_db.map(preprocess_1d)

    return train_db, test_db


def preprocess_2d(x, y):
    """
    (32,40,40,101)->(1280,40,101)
    (40,101)->(40,101,1)
    """
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [-1, 40, 101])
    # h40,w101,信道1
    # x = tf.reshape(x, (-1, 40, 101, 1))
    # 将标签转成one-hot形式
    y = tf.reshape(y, [-1])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, 2)

    return x, y


def load_data_2d(batch_size=128):
    '''
    加载数据集,返回一个tf.data.Dataset对象
    '''
    data = np.load('outfile1.npy')
    valence_labels = np.load('outfile2.npy')

    # arousal_labels = np.load('outfile3.npy')

    # 划分测试集与训练集
    data_train, data_test, valence_labels_train, valence_labels_test = train_test_split(data, valence_labels,
                                                                                        test_size=0.2,
                                                                                        random_state=seed)

    # 封装成tf.data.Dataset数据集对象
    train_db = tf.data.Dataset.from_tensor_slices((data_train, valence_labels_train))
    # 设置每次迭代的mini_batch大小
    train_db = train_db.batch(batch_size)
    # 对数据进行预处理
    train_db = train_db.map(preprocess_2d)
    # 打乱数据顺序
    train_db = train_db.shuffle(10000)

    # 封装数据集对象
    test_db = tf.data.Dataset.from_tensor_slices((data_test, valence_labels_test))
    # 设置mini_batch
    test_db = test_db.batch(batch_size)
    # 进行数据预处理
    test_db = test_db.map(preprocess_2d)

    return train_db, test_db
