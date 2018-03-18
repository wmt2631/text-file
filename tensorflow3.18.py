import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1) #设置图级随机seed
np.random.seed(1)

# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)； np.newaxis 创建一个新的维度 
noise = np.random.normal(0, 0.1, size=x.shape)  # 正态分布 中心 标准差
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise

# plot data
plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)     # input x   placeholder 占位符
tf_y = tf.placeholder(tf.float32, y.shape)     # input y

# neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu)          # hidden layer dense()构造一个致密层。将神经元数量和激活函数作为参数
output = tf.layers.dense(l1, 1)                     # output layer


loss = tf.losses.mean_squared_error(tf_y, output)   # compute cost 均方误差误差 X是真实数据 Y是预测数据 共有N个 那么 MSE = sum((X-Y).^2)/N
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)
sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph 初始化
plt.ion()   # something about plotting 打开交互模式


for step in range(100):
    # train and net output
    _, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
    if step % 5 == 0:

        # plot and show learning process
        plt.cla() #清空图
        plt.scatter(x, y) #点
        plt.plot(x, pred, 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1) #暂停0.1秒

plt.ioff() #关闭交互模式
plt.show()
