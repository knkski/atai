"""Moves blacklisted fonts into a separate blacklisted/ directory.

Useful for verifying that the blacklisted fonts are correctly blacklisted, and not
actually good fonts.
"""

import os
import shutil


def A(i):
    return chr(ord('A') + int(i))


with open('blacklist.txt') as f:
    blacklist = [l.strip() for l in f.readlines()]

shutil.rmtree('blacklisted', ignore_errors=True)
os.mkdir('blacklisted')

for i in range(10):
    os.mkdir("blacklisted/%s" % A(i))

for item in blacklist:
    for i in range(10):
        try:
            shutil.copy(f'notMNIST_large/{A(i)}/{item}', f'blacklisted/{A(i)}/{item}')
        except FileNotFoundError:
            # Some fonts are missing particular characters
            pass
