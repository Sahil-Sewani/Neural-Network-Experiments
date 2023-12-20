# PART 1 THE MODEL

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import math

from tensorflow import keras


lr = 0.001
n_epochs = 100

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


y_train = y_train.flatten();
y_test = y_test.flatten();

n_classes = 10
n_features = x_train[0,...].size
n_train = x_train.shape[0]
batch_size = 128

x_train, x_test = x_train.reshape([-1, n_features]), x_test.reshape([-1, n_features])

x_train = (x_train / 255.).astype(dtype=np.float32)
x_test = (x_test / 255).astype(dtype=np.float32)



n_classes = 10
n_features = x_train[0,...].size
n_train = x_train.shape[0]
batch_size = 128

#citerion= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#optimizer = tf.keras.optimizers.Adam(lr)

model = keras.Sequential(
    [
        keras.layers.Dense(512, input_shape=(n_features,), activation="relu", name="layer1"),
        keras.layers.Dense(128, activation="relu", name="layer2"),
        # we need to add softmax to last layer,
        # because model.compile(loss='sparse_categorical_crossentropy'
        # assumes from_logits=False, i.e., takes probabilities
        keras.layers.Dense(n_classes, activation="softmax", name="layer3"),
    ]
)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              #optimizer=optimizer,
              #loss=citerion,
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Initial test error:", 1 - test_acc)

# Train the model
history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Final test error:", 1 - test_acc)

# Plot the model
epochs = range(1, n_epochs + 1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], 'blue', label='Training accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'red', label='Test accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, 1 - np.array(history.history['accuracy']), 'blue', label='Training error')
plt.plot(epochs, 1 - np.array(history.history['val_accuracy']), 'red', label='Test error')
plt.title('Training and Validation Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend()
plt.show()

# MNIST code is to test that it's working correctly before testing on CIFAR10

## The above code runs the classification model on the MNIST database (LINE 16). You need to copy the code into your Jupyter
## notebook and comment out LINE 16 and uncomment LINE 17 to run the model on the CIFAR10 database.
## You can have both sets of code in you Jupyter notebook if you want (in different code boxes)
## MNIST code is to test that it's working correctly before testing on CIFAR10

#######################################################

# PART 2 THE EXPERIMENTS

# CHANGING NUMBERS OF LAYERS

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import math

from tensorflow import keras


lr = 0.001
n_epochs = 100

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


y_train = y_train.flatten();
y_test = y_test.flatten();

n_classes = 10
n_features = x_train[0,...].size
n_train = x_train.shape[0]
batch_size = 128

x_train, x_test = x_train.reshape([-1, n_features]), x_test.reshape([-1, n_features])

x_train = (x_train / 255.).astype(dtype=np.float32)
x_test = (x_test / 255).astype(dtype=np.float32)



n_classes = 10
n_features = x_train[0,...].size
n_train = x_train.shape[0]
batch_size = 128

#citerion= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#optimizer = tf.keras.optimizers.Adam(lr)

"""model = keras.Sequential(
    [
        keras.layers.Dense(512, input_shape=(n_features,), activation="relu", name="layer1"),
        keras.layers.Dense(128, activation="relu", name="layer2"),
        # we need to add softmax to last layer,
        # because model.compile(loss='sparse_categorical_crossentropy'
        # assumes from_logits=False, i.e., takes probabilities
        keras.layers.Dense(n_classes, activation="softmax", name="layer3"),
    ]
)"""

layers = [0, 1, 2, 3, 4, 5]  # list of layers to experiment with. 0 hidden layers to 5
colors = ["red", "green", "blue", "cyan", "magenta", "yellow"]  # colors to separate each layer

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

for layer, color in zip(layers, colors):
    model = keras.Sequential()
    if layer == 0:
        model.add(keras.layers.Dense(n_classes, input_shape=(n_features,), activation="softmax"))
    else:
        model.add(keras.layers.Dense(512, input_shape=(n_features,), activation="relu"))
        for i in range(layer):
            if i == layer - 1:
                model.add(keras.layers.Dense(n_classes, activation="softmax"))
            else:  # if not last layer, add more layers
                model.add(keras.layers.Dense(256 // (2 ** i), activation="relu"))


    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  #optimizer=optimizer,
                  #loss=citerion,
                  metrics=['accuracy'])

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Initial test error:", 1 - test_acc)

    # Train the model
    history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Final test error:", 1 - test_acc)

    # Plot the model
    test_errors = [1 - acc for acc in history.history['val_accuracy']]
    train_errors = [1 - acc for acc in history.history['accuracy']]

    axes[0].plot(train_errors, linestyle='-', color=color, label=f'{layer} Layers')
    axes[1].plot(test_errors, linestyle='-', color=color, label=f'{layer} Layers')

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Error')
axes[0].set_title('Training Error for Different Numbers of Layers over Epochs')
axes[0].legend()

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Test Error')
axes[1].set_title('Test Error for Different Numbers of Layers over Epochs')
axes[1].legend()

plt.tight_layout()
plt.show()

#######################################################

# CHANGING ACTIVATION FUNCTION


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import math

from tensorflow import keras


lr = 0.001
n_epochs = 100

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


y_train = y_train.flatten()
y_test = y_test.flatten()

n_classes = 10
n_features = x_train[0,...].size
n_train = x_train.shape[0]
batch_size = 128

x_train, x_test = x_train.reshape([-1, n_features]), x_test.reshape([-1, n_features])

x_train = (x_train / 255.).astype(dtype=np.float32)
x_test = (x_test / 255).astype(dtype=np.float32)



n_classes = 10
n_features = x_train[0,...].size
n_train = x_train.shape[0]
batch_size = 128

#citerion= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#optimizer = tf.keras.optimizers.Adam(lr)

"""model = keras.Sequential(
    [
        keras.layers.Dense(512, input_shape=(n_features,), activation="relu", name="layer1"),
        keras.layers.Dense(128, activation="relu", name="layer2"),
        # we need to add softmax to last layer,
        # because model.compile(loss='sparse_categorical_crossentropy'
        # assumes from_logits=False, i.e., takes probabilities
        keras.layers.Dense(n_classes, activation="softmax", name="layer3"),
    ]
)"""

activation_function = ["linear", "tanh", "relu"]  # list of activation functions
colors = ["red", "green", "blue"]  # colors to separate each function

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

for activation, color in zip(activation_function, colors):
    model = keras.Sequential()
    model.add(keras.layers.Dense(512, input_shape=(n_features,), activation=activation))
    model.add(keras.layers.Dense(256, activation=activation))
    model.add(keras.layers.Dense(128, activation=activation))
    model.add(keras.layers.Dense(64, activation=activation))
    model.add(keras.layers.Dense(32, activation=activation))
    model.add(keras.layers.Dense(n_classes, activation="softmax"))

    print(model.summary())  # check model layers

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  #optimizer=optimizer,
                  #loss=citerion,
                  metrics=['accuracy'])

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Initial test error:", 1 - test_acc)

    # Train the model
    history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_test, y_test))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Final test error:", 1 - test_acc)

    # Plot the model
    test_errors = [1 - acc for acc in history.history['val_accuracy']]
    train_errors = [1 - acc for acc in history.history['accuracy']]

    axes[0].plot(train_errors, linestyle='-', color=color, label=f'{activation} Activation Function')
    axes[1].plot(test_errors, linestyle='-', color=color, label=f'{activation} Activation Function')

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training Error')
axes[0].set_title('Training Error for Different Activation Functions over Epochs')
axes[0].legend()

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Test Error')
axes[1].set_title('Test Error for Different Activation Functions over Epochs')
axes[1].legend()

plt.tight_layout()
plt.show()
