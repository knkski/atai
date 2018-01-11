"""Given a model and input images, predicts the character contained in each image

Accepts either a Keras `*.h5` or Tensorflow `*.pb` model. Expects input images
to be 28x28.

Example usage:

    python predict.py -m models/model.h5 notMNIST_large/*/a2F6b28udHRm.png

"""
import argparse

import numpy as np
import sys
import keras
from scipy.ndimage import imread
import tensorflow as tf


def predict_keras(model_filename, images):
    model = keras.models.load_model(model_filename)

    return np.argmax(model.predict(images), axis=1)


def predict_tf(model_filename, images):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")

    input_tensor = graph.get_tensor_by_name('prefix/zero_padding2d_1_input:0')
    output_tensor = graph.get_tensor_by_name('prefix/output_node0:0')

    predictions = np.zeros((images.shape[0],), dtype=int)

    # Tensorflow doesn't natively handle batching like Keras, so manually take care of that here
    batch_size = 256

    with tf.Session(graph=graph) as sess:
        for i in range(0, images.shape[0], batch_size):
            predictions[i:i + batch_size] = np.argmax(sess.run(output_tensor, feed_dict={
                input_tensor: images[i:i + batch_size, :, :, :],
            }), axis=1)

    return predictions


def predict(model_filename, filenames):
    if len(filenames) < 1:
        raise ValueError('No filenames passed in!')

    # Convert input images into numpy array
    images = np.zeros((len(filenames), 28, 28, 1))

    for i, image in enumerate(filenames):
        images[i, :] = imread(image).reshape(1, 28, 28, 1)

    # Run the model against the inputs, and convert from one-hot binary output into
    # a human-friendly character prediction.
    if model_filename.endswith('h5'):
        predictions = predict_keras(model_filename, images)
    elif model_filename.endswith('pb'):
        predictions = predict_tf(model_filename, images)
    else:
        print('Unknown model type!')
        sys.exit(1)

    predictions = [
        chr(ord('A') + prediction)
        for prediction in predictions
    ]

    print('The predicted letters for these images:')

    for filename, prediction in zip(filenames, predictions):
        print("%s: %s" % (filename, prediction))


# For use in final submission. Takes in a single image in the form of a 28x28 numpy array, and
# returns a prediction in the range 0-9
def classify(image):
    image = image.reshape((1, 28, 28, 1))
    return predict_tf('model.pb', image)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict character contained in an image')
    parser.add_argument('input_files', nargs='*', help='Images to predict on')
    parser.add_argument('-m', '--model', type=str, default='model.pb', help='The model to load')

    args = parser.parse_args()
    predict(args.model, args.input_files)
