import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# x = tf.constant([1, 2, 3])
# y = tf.constant([9, 8, 7])

# z = tf.add(x, y)
# z = x+y

# z = tf.subtract(x, y)
# z = x-y

# z = tf.divide(x, y)
# z = x/y

# z = tf.multiply(x, y)
# z = x*y
# z = tf.tensordot(x, y, axes=1)
# z = tf.reduce_sum(x * y, axis=0)
# z = x ** 5
x = tf.random.normal((2, 3))
y = tf.random.normal((3, 4))
z = tf.matmul(x, y)
# z = z@y
print(z)
