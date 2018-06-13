import os
import threading

import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt


IMAGE = 0
DOCUMENT = 1
DOC_PATH = 'documents//'
IMG_PATH = 'images//'
STRIP_SIZE = 10
IMG_MEAN = 0.403474
IMG_STD = 0.255687
DOC_MEAN = 0.946252
DOC_STD = 0.194717


def calc_rel_pixel_center(idx, tiles):
    row = (idx - 1) // tiles
    col = (idx - 1) % tiles

    step = 1 / tiles

    center_x = (col + 0.5) * step
    center_y = (row + 0.5) * step

    return center_x, center_y


tile_rel_center = {tiles: [calc_rel_pixel_center(i, tiles) for i in range(1, tiles ** 2 + 1)] for tiles in [2, 4, 5]}

tile_neighbors_lr = {tiles: set([(i, i+1) for i in range(0, tiles ** 2) if (i + 1) % tiles != 0]) for tiles in [2, 4, 5]}
tile_neighbors_ud = {tiles: set([(i, i + tiles) for i in range(0, tiles ** 2 - tiles)]) for tiles in [2, 4, 5]}

all_tiles = {tiles: set([(i, j) for i in range(tiles ** 2) for j in range(tiles ** 2) if i != j]) for tiles in [2, 4, 5]}


non_neighbor_tiles_lr = {tiles: list(all_tiles[tiles].difference(tile_neighbors_lr[tiles])) for tiles in [2, 4, 5]}
non_neighbor_tiles_ud = {tiles: list(all_tiles[tiles].difference(tile_neighbors_ud[tiles])) for tiles in [2, 4, 5]}


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
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def process_image(image, is_img=True, resize=None):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if resize is not None:
        image = cv2.resize(image, resize)
    image = image / 255.0
    if is_img:
        image = (image - IMG_MEAN) / IMG_STD
    else:
        image = (image - DOC_MEAN) / DOC_STD

    image = np.expand_dims(image, -1)

    return image


def get_images_from_path(path, is_img=True, resize=None, process=True):
    files = os.listdir(path)
    images = []
    for f in files:
        im = cv2.imread(path + f)
        if process:
            im = process_image(im, is_img, resize)
        images.append(im)

    return np.asarray(images)


def create_images_strip(im1, im2, is_left_right, pixels):
    if not is_left_right:
        im1 = np.rot90(im1, 1)
        im2 = np.rot90(im2, 1)

    new_image = np.concatenate([im1, im2], axis=1)
    new_image = new_image[:, (im1.shape[1] - pixels):(im1.shape[1] + pixels)]

    return new_image


def get_image_neighbors(images, is_left_right=True, pixels = 3):
    tiles = np.sqrt(images.shape[0])

    tile_pairs = tile_neighbors_lr[tiles] if is_left_right else tile_neighbors_ud[tiles]
    non_neighbor_pairs = non_neighbor_tiles_lr[tiles] if is_left_right else non_neighbor_tiles_ud[tiles]
    non_neighbor_pairs = shuffle(non_neighbor_pairs)
    #non_neighbor_pairs = shuffle(non_neighbor_pairs)

    returned_images = []
    labels = []

    for i, j in tile_pairs:
        new_image = create_images_strip(images[i], images[j], is_left_right, pixels)
        returned_images.append(new_image)
        labels.append(1)

    for i, j in non_neighbor_pairs[:len(tile_pairs)]:
        new_image = create_images_strip(images[i], images[j], is_left_right, pixels)
        returned_images.append(new_image)
        labels.append(0)

    # while len(returned_images) < 2 * len(tile_pairs):
    #     i, j = np.random.randint(0, tiles, 2)
    #     if (i, j) not in tile_pairs:
    #         new_image = create_images_strip(images[i], images[j], is_left_right, pixels)
    #         returned_images.append(new_image)
    #         labels.append(0)

    return np.asarray(returned_images), np.asarray(labels)


def split_train_test_val(path, train_size=0.7, test_size=0.15, seed=42):
    valid_size = 1 - train_size - test_size
    assert valid_size > 0

    files = os.listdir(path)

    files = [path + file + '//' for file in files]

    train_files, other_files = train_test_split(files, train_size=train_size, random_state=seed)
    test_files, val_files = train_test_split(other_files, train_size=(test_size / (1 - train_size)), random_state=seed)

    return train_files, test_files, val_files


@threadsafe_generator
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
            process_image(im, (150, 150))
            images.append(im)
            batch_labels.append(label)

            if len(images) == 100:
                batch_labels = np.asarray(batch_labels)
                images = np.expand_dims(images, -1)
                yield images, batch_labels
                images = []
                batch_labels = []


@threadsafe_generator
def neighbors_gen(paths, pixels, is_left_right=True, is_img=True, randomize=False):

    if randomize:
        paths = shuffle(paths)

    while True:
        for path in paths:
            images = get_images_from_path(path, is_img, resize=None)
            images, labels = get_image_neighbors(images, is_left_right, pixels=pixels)
            yield images, labels


def centers_gen(paths, is_img=True, randomize=False):

    if randomize:
        paths = shuffle(paths)

    while True:
        for path in paths:
            images = get_images_from_path(path, is_img, resize=None)
            k = {4: 2, 16: 4, 25: 5}[len(images)]
            labels = tile_rel_center[k]
            labels1 = np.asarray([a for a, b in labels])
            labels2 = np.asarray([b for a, b in labels])
            yield images, [labels1, labels2]


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    points_true = []
    sizes_true = []

    points_false = []
    sizes_false = []

    X_img_train, X_img_test, X_img_val = split_train_test_val(DOC_PATH, 0.7, 0.15, seed=42)

    sum = 0.0
    count = 0

    for path in X_img_train:
        imgs = get_images_from_path(path)
        sum += np.sum(imgs)
        count += len(imgs) * imgs[0].shape[0] * imgs[0].shape[1]

    mean = sum / count

    sum = 0.0

    for path in X_img_train:
        imgs = get_images_from_path(path)
        sum += np.sum(np.square(imgs - mean))

    std = np.sqrt(sum / count)

    print("mean = %f, std = %f" % (mean, std))

    # for path in X_img_train[:500]:
    #     images = get_images_from_path(path, resize=None)
    #     images, labels = get_image_neighbors(images, True, 3)
    #
    #     for i in range(images.shape[0]):
    #         image = images[i][:, 0] - images[i][:, 1]
    #         size = image.shape[0]
    #         point = np.mean(np.absolute(image))
    #         if labels[i] == 1:
    #             points_true.append(point)
    #             sizes_true.append(size)
    #         else:
    #             points_false.append(point)
    #             sizes_false.append(size)
    #
    # print("")
