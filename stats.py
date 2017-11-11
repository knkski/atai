"""Calculates statistics for a given model and dataset."""

import numpy as np
import pandas as pd
import sys
import keras
from scipy.misc import imsave
import shutil
import os
from sklearn.metrics import confusion_matrix


def A(i):
    return chr(ord('A') + int(i))


def main(input_file):
    # Load model
    model = keras.models.load_model('model.h5')

    with np.load(input_file) as f:
        data = f['data'] / 255
        labels = f['labels']
        filenames = f['filenames']

    data = data.reshape(data.shape[0], 28, 28, 1)

    predictions = np.argmax(model.predict(data), axis=1)

    errors = labels != predictions

    df = pd.DataFrame(
        index=['Total'] + [chr(ord('A') + i) for i in range(10)],
        columns=['Errors', 'Total', 'Accuracy'],
    )

    df.iloc[0] = [sum(errors), len(errors), 1 - sum(errors) / len(errors)]

    for i in range(10):
        character_errors = labels[labels == i] != predictions[labels == i]

        df.iloc[1 + i] = [
            sum(character_errors),
            len(character_errors),
            1 - sum(character_errors) / (len(character_errors) or 1),
        ]

    print("Statistics:")
    print(df)

    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions, labels=[chr(ord('A') + i) for i in range(10)]))

    print("Saving wrongly-predicted images.")

    shutil.rmtree('errors', ignore_errors=True)
    os.mkdir('errors')

    for i in range(10):
        os.mkdir(f"errors/{A(i)}")

    for i, row in enumerate(data[errors]):
        char = A(labels[errors][i])
        original_filename = filenames[errors][i]
        predicted_char = A(predictions[errors][i])
        imsave(f"errors/{char}/{predicted_char}-{original_filename}", row.reshape(28, 28))

    print('Done saving images.')


if __name__ == '__main__':
    main(sys.argv[1])
