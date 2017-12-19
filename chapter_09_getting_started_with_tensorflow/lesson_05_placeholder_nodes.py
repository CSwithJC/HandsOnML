import tensorflow as tf
import numpy as np

"""Placeholder nodes don't perform any actual computation, they just
output the data you tell them to output at runtime. They are typically
used to pass the data to TensorFlow during training. If you don't specify
a value at runtime for a placeholder, you get an exception.
"""

# Specifying "None" for a dimension means "any size"
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5

with tf.Session() as sess:
    # A must have rank 2 (must be two-dimensional)
    # There must be three columns, or an exception will be raised
    # May have any number of rows
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

print(B_val_1)
print('\n')
print(B_val_2)
