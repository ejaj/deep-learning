import tensorflow as tf

tf = tf.compat.v1
tf.compat.v1.disable_v2_behavior()

x = tf.constant([[1, 2]])
neg_x = tf.negative(x)
print(neg_x)
