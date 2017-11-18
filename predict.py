import numpy as np
import sys
import keras
from scipy.ndimage import imread


def predict(filenames):
    # Load model
    model = keras.models.load_model('model.h5')

    if len(filenames) < 1:
        raise ValueError('No filenames passed in!')

    images = np.zeros((len(filenames), 28, 28, 1))

    for i, image in enumerate(images):
        images[i, :] = imread(image).reshape(1, 28, 28, 1)

    # Run the model against the inputs, and convert from one-hot binary output into
    # a human-friendly character prediction
    predictions = [
        chr(ord('A') + prediction)
        for prediction in np.argmax(model.predict(images), axis=1)
    ]

    print('The predicted letters for these images:')

    for filename, prediction in zip(filenames, predictions):
        print(f"{filename}: {prediction}")


if __name__ == '__main__':
    predict(sys.argv[1:])