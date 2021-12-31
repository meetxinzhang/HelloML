# import numpy as np
#
# a = [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]]
#
# b = [[1, 2, 3],
#      [4, 5, 6],
#      [7, 8, 9]]
#
# print(np.c_[a, b])
#
# print(np.r_[a, b])

# 这是一个识别"异或门"的BP神经网络
import tensorflow as tf
import numpy as np

a = 2
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([[0], [1], [1], [0]])


def initb(shape):
    return tf.Variable(tf.zeros(shape))


def initw(shape):
    return tf.Variable(tf.random_normal(shape, 0, 0.1))


with tf.name_scope('input'):
    x = tf.placeholder('float', [4, 2])
    y = tf.placeholder('float', [4, 1])

with tf.name_scope('L1'):
    W = initw([2, a])
    c = initb([1, a])
    # 矩阵c的维度会自动扩充
    f = tf.nn.relu(tf.matmul(x, W) + c)

with tf.name_scope('output'):
    w = initw([a, 1])
    y_ = tf.matmul(f, w)

with tf.name_scope('train'):
    # loss=-tf.reduce_sum(y*tf.log(y_))
    loss = tf.reduce_sum(tf.square(y - y_))
    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    #with tf.summary.FileWriter('./',sess.graph):
    	#...
    for i in range(600):
        print(sess.run(loss, {x: X, y: Y}))
        sess.run(train, {x: X, y: Y})
    print(sess.run([W, c, w]))

    # print(sess.run(tf.matmul(x, W), feed_dict={x: X, y: Y}))
    # print(sess.run(c, feed_dict={x: X, y: Y}))
