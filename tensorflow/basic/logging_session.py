import tensorflow as tf

tf = tf.compat.v1
tf.disable_v2_behavior()
x = tf.constant([[1., 2.]])
neg_x = tf.negative(x)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    result = session.run(neg_x)
print(result)
