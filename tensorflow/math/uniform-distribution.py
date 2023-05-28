import matplotlib.pyplot as plt
import tensorflow

tf = tensorflow.compat.v1
tf.compat.v1.disable_v2_behavior()
uniform = tf.random_uniform([100], minval=0, maxval=1, dtype=tf.float32)
with tf.Session() as session:
    print(uniform.eval())
    plt.hist(uniform.eval(), density=False)
    plt.show()

# Normal distribution
norm = tf.random_normal([100], mean=0, stddev=2)
with tf.Session() as sess:
    plt.hist(norm.eval(), normed=True)
    plt.show()
