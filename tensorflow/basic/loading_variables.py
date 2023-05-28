import tensorflow as tf

tf = tf.compat.v1
tf.disable_v2_behavior()

sess = tf.InteractiveSession()
spikes = tf.Variable([False] * 8, name='spikes')
saver = tf.train.Saver()

saver.restore(sess, "./spikes.ckpt")
print(spikes.eval())
sess.close()
