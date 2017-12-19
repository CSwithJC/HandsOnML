import tensorflow as tf


"""TensorFlow lets you stay DRY (Don't Repeat Yourself), using functions you can create,
such as relu(X) here.
"""

def relu(X):
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, 0., name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

# Create five relus and compute their sum
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
