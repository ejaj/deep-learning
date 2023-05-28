# x = 1
# y = x + 9
# print(y)

import tensorflow as tf

with tf.compat.v1.Session() as session:
    x = tf.constant(1, name='x')
    y = tf.Variable(x + 9, name='y')

    session.run(tf.compat.v1.global_variables_initializer())
    print(session.run(y))
