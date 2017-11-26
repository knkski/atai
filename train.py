"""Trains a deep neural network against the notMNIST dataset.

Adapted from the MNIST Keras example here:

https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

NOTE: This model requires a GPU to the model in a reasonable amount of time.
For a smaller model that trains better on a CPU, see [train.ipynb]
"""
from __future__ import division
import argparse
import os
import sys
from datetime import datetime

import keras
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dense, Dropout, Flatten, ZeroPadding2D, LeakyReLU, MaxPooling2D, BatchNormalization, ELU
from keras.models import Sequential
from sklearn.model_selection import train_test_split


def main(input_file, log_dir):
    batch_size = 2048
    num_classes = 10
    epochs = 50

    now = datetime.now().strftime('%Y-%m-%d_%H-%M')

    # input image dimensions
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    with np.load(input_file) as f:
        data = f['data']
        labels = f['labels']

    x_train, x_test, y_train, y_test = train_test_split(data, labels)

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

    # This model achieves better accuracy than the one in [train.ipynb],
    # but takes much longer to train. Should not be run without access
    # to a beefy GPU
    model = Sequential([
        ZeroPadding2D((2, 2), input_shape=input_shape),

        # First layer of convolutional networks
        Conv2D(64, kernel_size=(3, 3), activation='linear'), LeakyReLU(0.001),
        Conv2D(64, kernel_size=(3, 3), activation='linear'), LeakyReLU(0.001),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        # Let's try another layer
        Conv2D(128, kernel_size=(3, 3), activation='linear'), LeakyReLU(0.001),
        Conv2D(128, kernel_size=(3, 3), activation='linear'), LeakyReLU(0.001),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        Flatten(),
        Dense(4096, activation='linear'), LeakyReLU(0.001),
        Dense(4096, activation='linear'), LeakyReLU(0.001),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    print("Model Summary:")
    print(model.summary())

    # Declare tensorboard callback
    tensorboard = TensorBoard(
        log_dir=os.path.join(log_dir, now),
        histogram_freq=0,
        write_graph=True,
        write_images=False,
    )

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
    model.save(os.path.join('models', '%s.h5' % now))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on notMNIST dataset.')
    parser.add_argument('input_file', type=str, help='Input file to train on')
    parser.add_argument('-l', '--log-dir', type=str, default='logs', help='Where to write tensorboard logs to')

    args = parser.parse_args()

    main(args.input_file, args.log_dir)
