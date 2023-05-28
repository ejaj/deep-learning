import tensorflow as tf

with tf.compat.v1.Session() as session:
    a = tf.compat.v1.constant(10, name='a')
    b = tf.compat.v1.constant(20, name='b')
    y = tf.compat.v1.Variable(a + b * 2, name='y')
    merged = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter("tensorflowlogs", session.graph)
    session.run(tf.compat.v1.global_variables_initializer())
    print(session.run(y))
