import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

"""Same as before, but let's use Batch Gradient Descent instead of the
Normal Equation.
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

# Normalize the input feature vector or training may be much slower.
scaler = StandardScaler()
scaler.fit(housing_data_plus_bias)
scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)

# MANUALLY COMPUTING THE GRADIENTS:
n_epochs = 1000
learning_rate = 0.01

# Create X and y
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

# Theta are initially a lot of random variables between -1 and 1
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

# Matrix Multiplication
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)

# Assign function creates a node that will assign a new value to a variable.
# In this case, it implements Batch Gradient Descent step theta = theta - learningRate*gradient*mse
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval(session=sess)

print("Best theta is: ")
print(best_theta)


"""Same as before, but using TensorFlow's gradient function to calculate the gradient
instead of having to do it manually.
"""

# Theta are initially a lot of random variables between -1 and 1
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

# Matrix Multiplication
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

# Gradient function:
gradients = tf.gradients(mse, [theta])[0]

# Assign function creates a node that will assign a new value to a variable.
# In this case, it implements Batch Gradient Descent step theta = theta - learningRate*gradient*mse
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval(session=sess)

print("Best theta is: ")
print(best_theta)
