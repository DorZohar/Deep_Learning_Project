from doc_image_classifier import DocImageClassifier
from neighbors_classifier import NeighborsClassifier
from tile_pos_regressor import TilePosRegressor
import numpy as np
import utils
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def get_inference_log_prob(labels, left_right_probs, up_down_probs, k):
    labels = labels.flatten()
    left_right_pairs = [(i, i + 1) for i in range(0, len(labels)) if (i + 1) % k != 0]
    up_down_pairs = [(i, i + k) for i in range(0, len(labels) - k)]

    log_prob = 0.0
    for left, right in left_right_pairs:
        log_prob += np.log(left_right_probs[(labels[left], labels[right])])

    for up, down in up_down_pairs:
        log_prob += np.log(up_down_probs[(labels[up], labels[down])])

    return log_prob


def infer(left_right_probs, up_down_probs, k):

    best_labels = None
    best_log_prob = float('-inf')

    for i in range(1, k*k + 1):
        left_right_probs[(0, i)] = 1.0
        up_down_probs[(0, i)] = 1.0

    for p in range(1, k*k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[0][0] = p
        total_log_prob = 0.0
        available = set(range(1, k*k + 1))
        available.remove(p)
        for i in range(k):
            for j in range(k):
                if i == j == 0:
                    continue

                best_prob = 0.0
                best_piece = 0
                for piece in available:
                    prob = up_down_probs[(labels[i - 1][j], piece)] * left_right_probs[(labels[i][j - 1], piece)]
                    if prob >= best_prob:
                        best_prob = prob
                        best_piece = piece
                labels[i][j] = best_piece
                available.remove(best_piece)
                total_log_prob += np.log(up_down_probs[(labels[i - 1][j], piece)]) + \
                                  np.log(left_right_probs[(labels[i][j - 1], piece)])
        #log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        if total_log_prob > best_log_prob:
            best_log_prob = total_log_prob
            best_labels = labels

    return list(best_labels.flatten())


def dist(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def infer_by_pos(pred_positions, k):

    weights = np.zeros((k ** 2, k ** 2))
    actual_positions = utils.tile_rel_center[k]

    for i in range(k ** 2):
        for j in range(k ** 2):
            weights[i][j] = dist(pred_positions[i], actual_positions[j])

    row_ind, col_ind = linear_sum_assignment(weights)

    labels = np.zeros((k**2))
    labels[row_ind] = col_ind

    return labels


class Solver:

    def __init__(self, is_regression=False):
        self.is_regression = is_regression
        self.doc_image_classifier = DocImageClassifier(load_model=True)
        self.doc_lr_neighbors_classifier = NeighborsClassifier(is_docs=False, is_left_right=True, load_model=True) # NeighborsClassifier(is_docs=True, is_left_right=True, load_model=True)
        self.doc_ud_neighbors_classifier = NeighborsClassifier(is_docs=False, is_left_right=False, load_model=True) # NeighborsClassifier(is_docs=True, is_left_right=False, load_model=True)
        self.img_lr_neighbors_classifier = NeighborsClassifier(is_docs=False, is_left_right=True, load_model=True)
        self.img_ud_neighbors_classifier = NeighborsClassifier(is_docs=False, is_left_right=False, load_model=True)

        self.tile_pos_regression = TilePosRegressor(load_model=True)

    def predict_neigbors(self, images):
        left_right_probs = dict()
        up_down_probs = dict()
        k = {4: 2, 16: 4, 25: 5}[len(images)]

        is_doc = np.round(self.doc_image_classifier.predict(images)) == utils.DOCUMENT
        lr_classifier = self.doc_lr_neighbors_classifier if is_doc else self.img_lr_neighbors_classifier
        ud_classifier = self.doc_ud_neighbors_classifier if is_doc else self.img_ud_neighbors_classifier

        for i1, img1 in enumerate(images, 1):
            for i2, img2 in enumerate(images, 1):
                if i1 != i2:
                    left_right_probs[(i1, i2)] = lr_classifier.predict(img1, img2)[0][0]
                    up_down_probs[(i1, i2)] = ud_classifier.predict(img1, img2)[0][0]

        return infer(left_right_probs, up_down_probs, k)

    def predict_regression(self, images):
        predictions = self.tile_pos_regression.predict(images)
        k = {4: 2, 16: 4, 25: 5}[len(images)]

        predictions = predictions # preprocess

        return infer_by_pos(predictions, k)

    def predict(self, images):
        if self.is_regression:
            return self.predict_regression(images)
        else:
            return self.predict_neigbors(images)

    def evaluate(self, images, labels):

        predictions = self.predict(images)
        acc = np.mean(np.asarray(predictions) == np.asarray(labels))

        return acc


if __name__ == '__main__':
    X_img_train, X_img_test, X_img_val = utils.split_train_test_val(utils.IMG_PATH, 0.7, 0.15, seed=42)
    solver = Solver()

    sum = 0
    count = 0
    for path in X_img_test:
        images = list(utils.get_images_from_path(path, True))
        tiles = len(images)
        count += tiles
        labels = list(range(1, tiles + 1))
        images, labels = shuffle(images, labels)
        acc = solver.evaluate(images, labels)
        print("%d tiles, %f acc" % (tiles, acc))
        sum += acc * tiles

    print("Overall accuracy: %f" % (sum / count))
