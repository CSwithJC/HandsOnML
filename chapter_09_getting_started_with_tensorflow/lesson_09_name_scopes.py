import tensorflow as tf
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

"""When the model becomes more complex, its a good idea to create name scopes
to group related nodes.
"""

housing = fetch_california_housing()
m, n = housing.data.shape

# Add extra bias input feature (x0 = 1) to all training instances
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# These are now placeholder nodes:
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
XT = tf.transpose(X)

# Normalize the input feature vector or training may be much slower.
scaler = StandardScaler()
scaler.fit(housing_data_plus_bias)
scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)

# Theta are initially a lot of random variables between -1 and 1
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

# Matrix Multiplication
y_pred = tf.matmul(X, theta, name="predictions")

# NAME SCOPE:
# Define error and mse using name scope called "loss"
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")


# Gradient function:
n_epochs = 1000
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

batch_size = 100
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = housing.target.reshape(-1, 1)[indices]

    return X_batch, y_batch


with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):

        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()


print("Best theta is: ")
print(best_theta)

# The mse and error nodes now are inside the loss namespace:
print(error.op.name)
print(mse.op.name)
