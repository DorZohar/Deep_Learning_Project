import cv2
import os
import numpy as np

Xa = []
Xb = []
y = []

IM_DIR = os.path.join(os.path.dirname(__file__), 'images_source')
OUTPUT_DIR = "images"
files = os.listdir(IM_DIR)

# update this number for 4X4 crop 2X2 or 5X5 crops.
# tiles_per_dim = 4


for f in files:
    im = cv2.imread(os.path.join(IM_DIR, f))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    height = im.shape[0]
    width = im.shape[1]
    for tiles_per_dim in [2, 4, 5]:
        frac_h = height // tiles_per_dim
        frac_w = width // tiles_per_dim
        i = 0
        path = os.path.join(OUTPUT_DIR, f[:-4], '_{}'.format(tiles_per_dim))
        # path = OUTPUT_DIR + f[:-4] + "_{}\\".format(tiles_per_dim)
        os.makedirs(path)
        for h in range(tiles_per_dim):
            for w in range(tiles_per_dim):
                crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
                cv2.imwrite(os.path.join(path, '{}.jpg'.format(i)), crop)
                i = i + 1
