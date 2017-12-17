import tensorflow as tf

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())  # 10
    print(z.eval())  # 15

"""In the code above, TensorFlow detects that y depends on w,
which depends on x, so it evaluates w first, then x, then y. However,
the values for x and w are not reused, so it is evaluating w and x twice.

To evaluate y and z efficiently, without evaluating w and x twice, you must
ask TensorFlow to evaluate both y and z in just one graph run, as follows:
"""

with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)  # 10
    print(z_val)  # 15

"""In single-process TensorFlow, multiple sessions do not share any state, even
if they reuse the same graph. In distributed TensorFlow, variable state is stored
on the servers, not in the sessions, so multiple sessions can share the same variables.
"""
