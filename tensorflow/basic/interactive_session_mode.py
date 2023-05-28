import tensorflow as tf

tf = tf.compat.v1
tf.compat.v1.disable_v2_behavior()

sess = tf.InteractiveSession()
x = tf.constant([[1., 2.]])
neg_x = tf.negative(x)
result = neg_x.eval()
print(result)
