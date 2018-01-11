import base64
import glob

import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageOps
from scipy.ndimage import imread


with open('blacklist.txt') as f:
    blacklist = set(l.strip() for l in f.readlines())


for font_file in glob.glob('fonts/*.ttf'):
    for char_code in range(10):
        char = chr(ord('A') + char_code)

        # Encode the font name as base64, like the dataset does
        b64_font_name = base64.b64encode(font_file.split('/')[-1].encode())

        # Create new image and draw a single letter onto it. Written as much larger
        # than 28x28, so that we can crop and resize properly
        image = Image.new('L', (64, 64))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_file, 64)
        draw.text((0, 0), char, font=font, fill='white')

        # Crop and resize letter, keeping aspect ratio
        image = image.crop(image.getbbox())
        max_d = max(image.size)
        new_dimensions = tuple(int((max_d - d) / 2) for d in image.size)
        image = ImageOps.expand(image, new_dimensions)
        image = image.resize((28, 28), resample=Image.LANCZOS)

        with open('notMNIST_large/%s/%s.png' % (char, b64_font_name.decode('utf-8')), 'wb') as f:
            image.save(f)


for size in ['small', 'large']:
    print('Converting notMNIST_%s dataset into numpy arrays' % size)

    dirs = glob.glob('notMNIST_%s/*' % size)

    data = np.zeros((0, 28, 28))
    labels = np.zeros((0,))
    filenames = np.array([], dtype=str)

    for d in sorted(dirs):
        print('Converting directory %s' % d)

        # Get list of images, filtering out blacklisted ones. The image filename
        # will look like `notMNIST_large/A/foo.png`, and the blacklist will just
        # have `foo.png`, so we grab the last filepath segment before comparing.
        images = [
            i
            for i in glob.glob(d + '/*.png')
            if i.split('/')[-1] not in blacklist
        ]

        # Preallocate memory for the image data
        letter_data = np.zeros((2 * len(images), 28, 28))

        # Load the image data into the numpy array. Some images are broken
        # (0 bytes), and we just skip them. It shouldn't be enough images
        # to bias the learning algorithms.
        for j, image in enumerate(images):
            try:
                im = imread(image)
                letter_data[2 * j, :, :] = im
                letter_data[2 * j + 1, :, :] = 255 - im
            except OSError:
                print("Skipping loading of %s." % image)

        # Create a 1-D array of numeric labels for the processed images. The
        # labels start at 0 for A and end at 9 for J.
        labels = np.append(labels, np.full(letter_data.shape[0], ord(d[-1]) - ord('A')))

        data = np.append(data, letter_data, axis=0)

        fnames = [i.split('/')[-1] for i in images]

        filenames = np.append(filenames, list(zip(fnames, [f.split('.')[0] + '_inv.png' for f in fnames])))

    print('Found %s records for notMNIST_%s dataset. Saving...' % (data.shape[0], size))

    np.savez('notmnist_%s.npz' % size, data=data, labels=labels, filenames=filenames)
