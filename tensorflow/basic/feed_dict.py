import tensorflow as tf

tf.compat.v1.disable_eager_execution()
a = tf.compat.v1.placeholder(tf.compat.v1.int32)
b = tf.compat.v1.placeholder(tf.compat.v1.int32)
y = tf.compat.v1.multiply(a, b)

sess = tf.compat.v1.Session()

print(sess.run(y, feed_dict={a: 3, b: 5}))
