import numpy as np

tensor_2d = np.array([(1, 2, 3, 4), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)])
print(tensor_2d)
print(tensor_2d[3][3])
print(tensor_2d[0:2, 0:2])
import tensorflow as tf

with tf.compat.v1.Session() as sess:
    tf_tensor = tf.compat.v1.convert_to_tensor(tensor_2d, dtype=tf.compat.v1.float64)
    print(sess.run(tf_tensor))
