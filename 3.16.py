import tensorflow as tf
import numpy as np

'''x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)  # 激活
for step in range(201):
    sess.run(train)
    if step & 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
'''
'''matrik1 = tf.constant([[3, 3]])
matrik2 = tf.constant([[2], [2]])
product = tf.matmul(matrik1, matrik2)  # 乘法

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()'''

