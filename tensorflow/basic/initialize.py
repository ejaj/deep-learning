import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf

# x = tf.constant(8)
# x = tf.constant(8, shape=(1, 1))
# x = tf.constant(3, shape=(1, 1), dtype=tf.float32)

# x = tf.constant([[1, 2, 3], [4, 5, 6]])
# x = tf.ones((3,3))
# x = tf.zeros((2,3))
# x = tf.eye(3)
# x = tf.random.normal((3, 3), mean=0, stddev=1)
# x = tf.random.uniform((1,3), minval=0, maxval=1)
# x = tf.range(9)
x = tf.range(start=1, limit=10, delta=2)
x = tf.cast(x, dtype=tf.float64)
# tf.float(16,6,7) #cast
# tf.int(3,3,3) cast
# tf.bool cast
print(x)
