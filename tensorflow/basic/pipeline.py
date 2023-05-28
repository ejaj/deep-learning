import tensorflow as tf
import numpy as np

tf = tf.compat.v1
tf.compat.v1.disable_v2_behavior()

x_input = np.random.sample((1, 2))
print(x_input)
x = tf.placeholder(tf.float32, shape=[1, 2], name="X")
dataset = tf.data.Dataset.from_tensors(x)
iterator = dataset.make_initializable_iterator()
get_next = iterator.get_next()
with tf.Session() as sess:
    # feed the placeholder with data
    sess.run(iterator.initializer, feed_dict={x: x_input})
    print(sess.run(get_next))
