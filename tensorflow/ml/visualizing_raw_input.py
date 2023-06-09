import numpy as np
import matplotlib.pyplot as plt
learning_rate = 0.01
training_epochs = 100
x_train = np.linspace(-1, 1, 101)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.33
plt.scatter(x_train, y_train)
plt.show()
