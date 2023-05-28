import tensorflow
import input_data
import matplotlib.pyplot as plt

tf = tensorflow.compat.v1
tf.compat.v1.disable_v2_behavior()

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer num features
n_hidden_2 = 256  # 2nd layer num features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# weights layer 1
h = tf.Variable(tf.random_normal([n_input, n_hidden_1]))

# bias layer 1
bias_layer_1 = tf.Variable(tf.random_normal([n_hidden_1]))

# layer 1
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, h), bias_layer_1))

# weights layer 2
w = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))

# bias layer 2
bias_layer_2 = tf.Variable(tf.random_normal([n_hidden_2]))

# layer 2
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, w), bias_layer_2))

# weights output layer
output = tf.Variable(tf.random_normal([n_hidden_2, n_classes]))

# bias output layer
bias_output = tf.Variable(tf.random_normal([n_classes]))

# output layer
output_layer = tf.matmul(layer_2, output) + bias_output

# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=y))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Plot settings
avg_set = []
epoch_set = []

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        avg_set.append(avg_cost)
        epoch_set.append(epoch + 1)

print("Training phase finished")
plt.plot(epoch_set, avg_set, 'o', label='MLP Training phase')
plt.ylabel('cost')
plt.xlabel('epoch')
plt.legend()
plt.show()
