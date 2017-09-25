"""Trains a deep neural network against the notMNIST dataset.

Adapted from the MNIST Keras example here:

https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
"""

from datetime import datetime
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
import keras
import numpy as np
import sys

GPU_ENABLED = any(d for d in device_lib.list_local_devices() if 'gpu' in d.name)

batch_size = 256
num_classes = 10
epochs = 40 if GPU_ENABLED else 10


# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

if len(sys.argv) > 1:
    with np.load(sys.argv[1]) as f:
        data = f['data']
        labels = f['labels']

    x_train, x_test, y_train, y_test = train_test_split(data, labels)
else:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Convert from integers to floats, so that we can have a range of 0-1
    # instead of 0-255
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

# Convert from 0-255 to 0-1
x_train /= 255
x_test /= 255

# Resize from (m, 28, 28) to (m, 28, 28, 1), since Keras always assumes the 4th
# dimension in case of RGB images, even if we're just doing B/W images.
x_train = x_train.reshape(x_train.shape[0], *input_shape)
x_test = x_test.reshape(x_test.shape[0], *input_shape)

print("Found %s records. Splitting into %s training and %s test records." % (
    x_train.shape[0] + x_test.shape[0],
    x_train.shape[0],
    x_test.shape[0],
))

# Convert from numeric labels to one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if GPU_ENABLED:
    # This model achieves better accuracy, but takes much longer to train. Should
    # not be run without access to a beefy GPU
    model = Sequential([
        # First layer of convolutional networks
        Conv2D(64, kernel_size=(3, 3), activation='linear', input_shape=input_shape),
        LeakyReLU(alpha=.001),
        Conv2D(64, kernel_size=(3, 3), activation='linear'),
        LeakyReLU(alpha=.001),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        # Let's try another layer
        Conv2D(128, kernel_size=(3, 3), activation='linear'),
        LeakyReLU(alpha=.001),
        Conv2D(128, kernel_size=(3, 3), activation='linear'),
        LeakyReLU(alpha=.001),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(4096, activation='linear'),
        LeakyReLU(alpha=.001),
        Dense(4096, activation='linear'),
        LeakyReLU(alpha=.001),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
else:
    # This is a simpler model that can easily be trained on a laptop
    model = Sequential([
        Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print("Model Summary:")
print(model.summary())

# Declare tensorboard callback
tensorboard = TensorBoard(log_dir='./logs/%s' % datetime.now().strftime('%Y-%m-%d_%H:%M'),
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

# Train model on training set
try:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tensorboard])
except KeyboardInterrupt:
    print("\n\nCaught KeyboardInterrupt, stopping training!")

# See how well it did on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Save model for later usage
model.save('model.h5')
