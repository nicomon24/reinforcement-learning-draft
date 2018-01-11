import numpy as np
import tensorflow as tf

class SGDRegressor:
    def __init__(self, D):
        print("Using TF SGDRegressor")
        lr = 10e-2
        self.w = tf.Variable(tf.random_normal(shape=(D, 1)), name='w')
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')
        # Matmul with reshape (because w is not good as 1D array)
        self.Y_hat = tf.reshape(tf.matmul(self.X, self.w), [-1])
        self.loss = tf.reduce_sum(tf.square(self.Y - self.Y_hat))
        # Define ops
        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(self.loss)
        self.predict_op = self.Y_hat
        # Define init
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

    def predict(self, X):
        return self.session.run(self.predict_op, feed_dict={self.X: X})
