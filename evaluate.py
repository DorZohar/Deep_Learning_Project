import os
import numpy as np
import cv2
import utils
from solver import Solver


def predict(images):
    labels = []

    solver = Solver()

    # process image & predict if it's an image or doc

    images = utils.process_images(images)
    is_image = np.round(solver.doc_image_classifier.predict(images)) == utils.IMAGE
    images = utils.normalize_images(images, is_img=is_image)

    # predict labels

    labels = solver.predict(images, is_image)

    return labels


def evaluate(file_dir='example/'):
    files = os.listdir(file_dir)
    files.sort()
    images = []
    for f in files:
        im = cv2.imread(file_dir + f)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        images.append(im)

    Y = predict(images)
    return Y


