import tensorflow as tf
sess = tf.InteractiveSession()
#matrix(2X3) matrix(3X1)
w1 = tf.Variable(tf.random_normal([2,3],mean=1.0, stddev=1.0))
w2 = tf.Variable(tf.random_normal([3,1],mean=1.0, stddev=1.0))
#vector(1X2)
x = tf.constant([[0.7, 0.9]])

#new version init method
#tf.global_variables_initializer().run()
#old version init method
tf.initialize_all_variables().run()

#mult V X M
a = tf.matmul(x ,w1)
y = tf.matmul(a, w2)

print(y.eval())
