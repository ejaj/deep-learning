import tensorflow as tf

x = tf.range(9)

print(x)
x = tf.reshape(x, (3, 3))
print(x)

x = tf.transpose(x, perm=[1, 0])
print(x)
