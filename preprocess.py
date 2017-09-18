import glob
from scipy.ndimage import imread
import numpy as np

for size in ['small', 'large']:
    print('Converting notMNIST_%s dataset into numpy arrays' % size)

    dirs = glob.glob('notMNIST_%s/*' % size)

    data = np.zeros((0, 28, 28))
    labels = np.zeros((0,))

    for d in dirs:
        images = glob.glob(d + '/*.png')

        letter_data = np.zeros((len(images), 28, 28))

        # Load the image data into the nump array. Some images are broken
        # (0 bytes), and we just skip them.
        for j, image in enumerate(images):
            try:
                letter_data[j, :, :] = imread(image)
            except OSError:
                print("Skipping loading of %s." % image)

        # Create a 1-D array of numeric labels for the processed images. The
        # labels start a 0 for A and end at 9 for J.
        labels = np.append(labels, np.full(letter_data.shape[0], ord(d[-1]) - ord('A')))

        data = np.append(data, letter_data, axis=0)

    print('Found %s records for notMNIST_%s dataset. Saving...' % (data.shape[0], size))

    np.savez('notmnist_%s.npz' % size, data=data, labels=labels)
