import tensorflow as tf

# Create a computational graph, but doesn't actually do any operations.
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x * x * y + y + 2

# Open a TensorFlow session to evaluate the graph created above.

"""
# An Option:
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)

# Another Option:
with tf.Session() as sess:
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
"""

# Another option:
init = tf.global_variables_initializer()  # prepare an init node
with tf.Session() as sess:
    init.run()  # actually initialize all the variables
    result = f.eval()  # equivalent to calling tf.get_default_session().run(f)


# See results from graph and close session.
print(result)
sess.close()
