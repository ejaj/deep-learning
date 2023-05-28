import numpy as np

tensor_1d = np.array([1.3, 1, 4.0, 23.99])
# print(tensor_1d)
# print(tensor_1d[0])
# print(tensor_1d.ndim)
# print(tensor_1d.shape)
# print(tensor_1d.dtype)

# convert numpy to tensorflow

import tensorflow as tf

with tf.compat.v1.Session() as sess:
    tf_tensor = tf.compat.v1.convert_to_tensor(tensor_1d, dtype=tf.compat.v1.float64)
    print(sess.run(tf_tensor))
    print(sess.run(tf_tensor[0]))
    print(sess.run(tf_tensor[2]))
