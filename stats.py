"""Calculates statistics for a given model and dataset.

Example usage:

    python stats.py notmnist_large.npz

"""
from __future__ import division
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.misc import imsave
import shutil
import os

from predict import predict_keras, predict_tf


def A(i):
    return chr(ord('A') + int(i))


def main(input_file, model_file):
    with np.load(input_file) as f:
        data = f['data'] / 255
        labels = f['labels']
        filenames = f['filenames']

    data = data.reshape(data.shape[0], 28, 28, 1)

    if model_file.endswith('.h5'):
        predictions = predict_keras(model_file, data)
    elif model_file.endswith('.pb'):
        predictions = predict_tf(model_file, data)
    else:
        print('Unknown model type!')
        sys.exit(1)

    errors = labels != predictions

    df = pd.DataFrame(
        index=['Total'] + [A(i) for i in range(10)],
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

    print("Saving wrongly-predicted images.")

    shutil.rmtree('errors', ignore_errors=True)
    os.mkdir('errors')

    for i in range(10):
        os.mkdir("errors/%s" % A(i))

    for i, row in enumerate(data[errors]):
        char = A(labels[errors][i])
        original_filename = filenames[errors][i]
        predicted_char = A(predictions[errors][i])
        imsave("errors/%s/%s-%s" % (char, predicted_char, original_filename), row.reshape(28, 28))

    print('Done saving images.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show statistics for given dataset')
    parser.add_argument('input_file', type=str, help='Dataset to show statistics for')
    parser.add_argument('-m', '--model', type=str, default='model.pb', help='The model to load')

    args = parser.parse_args()
    main(args.input_file, args.model)
