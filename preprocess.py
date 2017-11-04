import glob
from scipy.ndimage import imread
import numpy as np


with open('blacklist.txt') as f:
    blacklist = set(l.strip() for l in f.readlines())


for size in ['small', 'large']:
    print('Converting notMNIST_%s dataset into numpy arrays' % size)

    dirs = glob.glob('notMNIST_%s/*' % size)

    data = np.zeros((0, 28, 28))
    labels = np.zeros((0,))

    for d in sorted(dirs):
        print(f'Converting directory {d}')

        # Get list of images, filtering out blacklisted ones. The image filename
        # will look like `notMNIST_large/A/foo.png`, and the blacklist will just
        # have `foo.png`, so we grab the last filepath segment before comparing.
        images = [
            i
            for i in glob.glob(d + '/*.png')
            if i.split('/')[-1] not in blacklist
        ]

        # Preallocate memory for the image data
        letter_data = np.zeros((len(images), 28, 28))

        # Load the image data into the numpy array. Some images are broken
        # (0 bytes), and we just skip them. It shouldn't be enough images
        # to bias the learning algorithms.
        for j, image in enumerate(images):
            try:
                letter_data[j, :, :] = imread(image)
            except OSError:
                print("Skipping loading of %s." % image)

        # Create a 1-D array of numeric labels for the processed images. The
        # labels start at 0 for A and end at 9 for J.
        labels = np.append(labels, np.full(letter_data.shape[0], ord(d[-1]) - ord('A')))

        data = np.append(data, letter_data, axis=0)

    print('Found %s records for notMNIST_%s dataset. Saving...' % (data.shape[0], size))

    np.savez('notmnist_%s.npz' % size, data=data, labels=labels)
