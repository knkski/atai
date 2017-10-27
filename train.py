"""Trains a deep neural network against the notMNIST dataset.

Adapted from the MNIST Keras example here:

https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

NOTE: This model requires a GPU to the model in a reasonable amount of time.
For a smaller model that trains better on a CPU, see [train.ipynb]
"""

import argparse
import sys
from datetime import datetime

import keras
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Dense, Dropout, Flatten, ZeroPadding2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split


def main(input_file):
    batch_size = 256
    num_classes = 10
    epochs = 40

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
        # First layer of convolutional networks
        ZeroPadding2D((2, 2), input_shape=input_shape),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        Dropout(0.5),

        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        Dropout(0.5),

        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        Dropout(0.5),

        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        Dropout(0.5),

        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        Dropout(0.5),

        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    print("Model Summary:")
    print(model.summary())

    # Declare tensorboard callback
    tensorboard = TensorBoard(log_dir='/output/%s' % datetime.now().strftime('%Y-%m-%d_%H-%M'),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on notMNIST dataset.')
    parser.add_argument('input_file', type=str, help='Input file to train on')

    args = parser.parse_args()

    main(args.input_file)
