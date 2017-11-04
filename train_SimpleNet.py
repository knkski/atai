"""Trains a SimpleNet against the notMNIST dataset.

Adapted from example here:

https://github.com/Coderx7/SimpleNet/blob/master/SimpNet_V1/Logs/MNIST/caffe_99.75.log

"""

import argparse
import sys
from datetime import datetime

import keras
import numpy as np
from keras.callbacks import TensorBoard
from keras import initializers
from keras.layers import Conv2D, Dense, Dropout, Flatten, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Sequential
from sklearn.model_selection import train_test_split


# YQ: change to batch size =100

def main(input_file, log_dir, model_output):
    batch_size = 100
    num_classes = 10
    epochs = 40

    # assign fine tuning values at beginning, easier for future adjustment
    output_size_1 = 64
    output_size_2 = 128
    output_size_3 = 256
    output_size_4 = 512
    output_size_5 = 2048

    kn_size_1 = (3, 3)
    kn_size_2 = (1, 1)

    p_size = (2, 2)

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

    # create model
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    # 1st Conv layer
    model.add(Conv2D(output_size_1,
                     kn_size_1, kernel_initializer=initializers.glorot_normal(seed=None)))
    BatchNormalization(axis=-1, momentum=0.95)
    model.add(Activation('relu'))

    # 2nd Conv layer
    model.add(ZeroPadding2D((1, 1)))
    # not sure why pad 1 in each convoluational layer...
    model.add(Conv2D(output_size_2,
                     kn_size_1, kernel_initializer=initializers.glorot_uniform(seed=None)))
    BatchNormalization(axis=-1, momentum=0.95)
    model.add(Activation('relu'))

    # 3rd Conv layer
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(output_size_2,
                     kn_size_1, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)))
    BatchNormalization(axis=-1, momentum=0.95)
    model.add(Activation('relu'))

    # 4th Conv layer
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(output_size_2,
                     kn_size_1, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)))
    BatchNormalization(axis=-1, momentum=0.95)
    model.add(Activation('relu'))
    # 1st pool layer after the 4th Conv layer
    model.add(MaxPooling2D(pool_size=p_size, strides=2))

    # 5th Conv layer
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(output_size_2,
                     kn_size_1, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None)))
    BatchNormalization(axis=-1, momentum=0.95)
    model.add(Activation('relu'))

    # 6th Conv layer
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(output_size_2,
                     kn_size_1, kernel_initializer=initializers.glorot_uniform(seed=None)))
    BatchNormalization(axis=-1, momentum=0.95)
    model.add(Activation('relu'))

    # 7th Conv layer
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(output_size_3,
                     kn_size_1, kernel_initializer=initializers.glorot_uniform(seed=None)))
    # 2nd pool layer after the 7th Conv layer before batch normalization
    model.add(MaxPooling2D(pool_size=p_size, strides=2))
    BatchNormalization(axis=-1, momentum=0.95)
    model.add(Activation('relu'))

    # 8th Conv layer
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(output_size_3,
                     kn_size_1, kernel_initializer=initializers.glorot_uniform(seed=None)))
    BatchNormalization(axis=-1, momentum=0.95)
    model.add(Activation('relu'))

    # 9th Conv layer
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(output_size_3,
                     kn_size_1, kernel_initializer=initializers.glorot_uniform(seed=None)))
    BatchNormalization(axis=-1, momentum=0.95)
    model.add(Activation('relu'))
    # 3rd pool layer after the 9th Conv layer
    model.add(MaxPooling2D(pool_size=p_size, strides=2))

    # 10th Conv layer
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(output_size_4,
                     kn_size_1, kernel_initializer=initializers.glorot_uniform(seed=None)))
    BatchNormalization(axis=-1, momentum=0.95)
    model.add(Activation('relu'))

    # 11th Conv layer
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(output_size_5,
                     kn_size_2, kernel_initializer=initializers.glorot_uniform(seed=None),
                     bias_initializer=initializers.Constant(value=0)))
    model.add(Activation('relu'))

    # 12th Conv layer
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(output_size_3,
                     kn_size_2, kernel_initializer=initializers.glorot_uniform(seed=None),
                     bias_initializer=initializers.Constant(value=0)))
    model.add(Activation('relu'))
    # 4th pool layer after the 12th Conv layer
    model.add(MaxPooling2D(pool_size=p_size, strides=2))

    # 13th Conv layer
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(output_size_3,
                     kn_size_1, kernel_initializer=initializers.glorot_uniform(seed=None),
                     bias_initializer=initializers.Constant(value=0)))
    model.add(Activation('relu'))
    # 5th pool layer after the 13th Conv layer
    model.add(MaxPooling2D(pool_size=p_size, strides=2))

    # 14th Fully connected layer
    model.add(Flatten())
    model.add(Dense(num_classes,
              kernel_initializer=initializers.glorot_uniform(seed=None),
              bias_initializer=initializers.Constant))
    model.add(Activation('softmax'))

    #  model.add(Dropout(0.5))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    print("Model Summary:")
    print(model.summary())

    # Declare tensorboard callback
    tensorboard = TensorBoard(
        log_dir='./logs/%s' % datetime.now().strftime('%Y-%m-%d_%H:%M'),
        histogram_freq=0,
        write_graph=True,
        write_images=False
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
    model.save('model.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model on notMNIST dataset.')
    parser.add_argument('input_file', type=str, help='Input file to train on')
    parser.add_argument('-l', '--log-dir', type=str, help='Tensorboard log dir')
    parser.add_argument('-m', '--model-output', type=str, help='Where to save the model file')

    args = parser.parse_args()

    main(args.input_file, args.log_dir, args.model_output)