import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# x_train = tf.convert_to_tensor(x_test) # alternative of reshape

# Sequential API, (very convenient and not very flexible, allow only one inout and one output)
model = keras.Sequential(
    [
        layers.Input(shape=(28 * 28)),
        layers.Dense(512, activation='relu', ),
        layers.Dense(256, activation='relu', name='my_layer'),
        layers.Dense(10),
    ]
)

# print(model.summary())

# another way to use of sequential api
# model = keras.Sequential()
# model.add(keras.Input(shape=(784)))
# model.add(layers.Dense(512, activation='relu'))
# print(model.summary())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(10))


# specific layer
# model = keras.Model(inputs=model.inputs,
#                     outputs=[model.get_layer('my_layer').output])
# feature = model.predict(x_train)
# print(feature.shape)
# all layers
# model = keras.Model(inputs=model.inputs,
#                     outputs=[layer.output for layer in model.layers])
#
# features = model.predict(x_train)
# for feature in features:
#     print(feature.shape)
#
# import sys
#
# sys.exit()
#
# model.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=keras.optimizers.Adam(learning_rate=0.001),
#     metrics=['accuracy']
# )
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
