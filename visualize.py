from scipy.ndimage import imread
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import sys


def visualize(model, image):
    image = imread(image).reshape(1, 28, 28, 1)

    activation_maps = [
        (layer.name, K.function([model.input, K.learning_phase()], [layer.output])([image, 0.])[0])
        for layer in model.layers
    ]

    print("The original input image:")
    plt.imshow(image.reshape(28, 28), interpolation='None', cmap='binary_r')
    plt.axis('off')
    plt.show()

    for layer_name, activation_map in activation_maps:
        print('Activation map for layer %s %s:' % (layer_name, activation_map.shape))

        if len(activation_map.shape) == 4:
            # Convert from vertically-stacked images to side-by-side in a line
            letters = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))

            # Rearrange those images into a square
            activations = np.vstack(np.split(letters, int(activation_map.shape[-1] ** 0.5), 1))

        elif len(activation_map.shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)

        else:
            raise Exception('Can\'t deal with shape %s' % (activation_map.shape,))

        plt.imshow(activations, interpolation='None', cmap='binary_r')
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    model = keras.models.load_model('model.h5')
    visualize(model, sys.argv[1])
