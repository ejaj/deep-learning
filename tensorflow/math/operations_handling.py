import tensorflow as tf
import numpy as np

tf.compat.v1.disable_v2_behavior()

matrix1 = np.array([(2, 2, 2), (2, 2, 2), (2, 2, 2)], dtype='int32')
matrix2 = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)], dtype='int32')
# print(matrix1)
# print(matrix2)

# transformed into a tensor data
tf_matrix1 = tf.compat.v1.constant(matrix1)
tf_matrix2 = tf.compat.v1.constant(matrix2)
matrix_3 = np.array([(2, 7, 2), (1, 4, 2), (9, 0, 2)], dtype='float32')
matrix_product = tf.compat.v1.multiply(matrix1, matrix2)
matrix_sum = tf.compat.v1.add(matrix1, matrix2)
matrix_det = tf.compat.v1.matrix_determinant(matrix_3)
with tf.compat.v1.Session() as sess:
    result1 = sess.run(matrix_product)
    result2 = sess.run(matrix_sum)
    result3 = sess.run(matrix_det)

print("Matrix product:")
print(result1)
print("Matrix sum:")
print(result2)
print("Matrix determinant:")
print(result3)
