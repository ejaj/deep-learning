import tensorflow as tf

tf = tf.compat.v1
tf.compat.v1.disable_v2_behavior()

X_1 = tf.placeholder(tf.float16, name="X_1")
X_2 = tf.placeholder(tf.float16, name="X_2")

add = tf.add(X_1, X_2, name="add")
sub = tf.subtract(X_1, X_2, name="sub")
multiply = tf.multiply(X_1, X_2, name="multiply")
div = tf.div(X_1, X_2, name="div")

with tf.Session() as session:
    add_result = session.run(add, feed_dict={X_1: [1, 2, 3], X_2: [4, 5, 6]})
    sub_result = session.run(sub, feed_dict={X_1: [1, 2, 3], X_2: [4, 5, 6]})
    mul_result = session.run(multiply, feed_dict={X_1: [1, 2, 3], X_2: [4, 5, 6]})
    div_result = session.run(div, feed_dict={X_1: [1, 2, 3], X_2: [4, 5, 6]})
    print(add_result)
    print(sub_result)
    print(mul_result)
    print(div_result)
