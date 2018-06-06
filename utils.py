import os
import threading

import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np


IMAGE = 0
DOCUMENT = 1
DOC_PATH = 'documents//'
IMG_PATH = 'images//'


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def get_images_from_path(path):
    files = os.listdir(path)
    images = []
    for f in files:
        im = cv2.imread(path + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = cv2.resize(im, (150, 150))
        im = im / 255.0
        images.append(im)

    return np.expand_dims(np.asarray(images), -1)


def split_train_test_val(path, train_size=0.7, test_size=0.15, seed=42):
    valid_size = 1 - train_size - test_size
    assert valid_size > 0

    files = os.listdir(path)

    files = [path + file + '//' for file in files]

    train_files, other_files = train_test_split(files, train_size=train_size, random_state=seed)
    test_files, val_files = train_test_split(other_files, train_size=(test_size / (1 - train_size)), random_state=seed)

    return train_files, test_files, val_files


def image_gen(paths, labels, randomize=False):

    if randomize:
        paths, labels = shuffle(paths, labels)

    images = []
    batch_labels = []

    while True:
        for path, label in zip(paths, labels):
            files = os.listdir(path)
            f = files[np.random.randint(len(files))]
            im = cv2.imread(path + f)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            im = cv2.resize(im, (150, 150))
            images.append(im)
            batch_labels.append(label)

            if len(images) == 100:
                batch_labels = np.asarray(batch_labels)
                images = np.asarray(images) / 255.0
                images = np.expand_dims(images, -1)
                yield images, batch_labels
                images = []
                batch_labels = []

