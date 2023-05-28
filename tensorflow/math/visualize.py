import matplotlib.image as mp_image
import matplotlib.pyplot as plt
import tensorflow
import numpy as np

tf = tensorflow.compat.v1
tf.compat.v1.disable_v2_behavior()

# Mandelbrot's set
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X + 1j * Y
c = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(c)
ns = tf.Variable(tf.zeros_like(c, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
zs_ = zs * zs + c
not_diverged = tf.abs(zs_) < 4
step = tf.group(zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)))
for i in range(200):
    step.run()
plt.imshow(ns.eval())
plt.show()
