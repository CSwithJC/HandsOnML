import tensorflow as tf
import numpy as np
from sklearn.datasets import  fetch_california_housing

"""TensorFlow operations (called ops) can take any number of inputs and
produce any number of outputs. Constants and variables take no input (these
are called source ops). The inputs and outputs are multidimensional arrays,
called Tensors.
"""

housing = fetch_california_housing()
m, n = housing.data.shape

# Add extra bias input feature (x0 = 1) to all training instances
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# Creates two TensorFlow constant nodes to hold this data and targets
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
# Reshape (-1, 1) here reshapes data to column vector.
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)

# The normal equation (theta = XT * X)^-1 * XT * y
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()

print(theta_value)
