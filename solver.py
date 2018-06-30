from doc_image_classifier import DocImageClassifier
from neighbors_classifier import NeighborsClassifier
from tile_pos_regressor import TilePosRegressor
import numpy as np
import utils
from sklearn.utils import shuffle
# import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from itertools import permutations
import operator

EPS = 0.00000000001


def get_inference_log_prob(labels, left_right_probs, up_down_probs, k):
    if type(labels) is not tuple:
        labels = labels.flatten()
    left_right_pairs = [(i, i + 1) for i in range(0, len(labels)) if (i + 1) % k != 0]
    up_down_pairs = [(i, i + k) for i in range(0, len(labels) - k)]

    log_prob = 0.0
    for left, right in left_right_pairs:
        log_prob += np.log(left_right_probs[(labels[left], labels[right])] + EPS)

    for up, down in up_down_pairs:
        log_prob += np.log(up_down_probs[(labels[up], labels[down])] + EPS)

    return log_prob


# def plot_img(images, id1, id2, vertical=False):
#     rows = images[id1 - 1].shape[0]
#     cols = images[id1 - 1].shape[1]
#     if vertical is False:
#         comb = np.zeros((rows, 2 * cols))
#         comb[:, :cols] = np.squeeze(images[id1 - 1])
#         comb[:, cols:] = np.squeeze(images[id2 - 1])
#     else:
#         comb = np.zeros((2 * rows, cols))
#         comb[:rows, :] = np.squeeze(images[id1 - 1])
#         comb[rows:, :] = np.squeeze(images[id2 - 1])
#     plt.imshow(comb, cmap='gray')
#     plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#     plt.title('image')
#     plt.show()


def greedy_top_left(left_right_probs, up_down_probs, k, images):
    best_labels = None
    best_log_prob = float('-inf')

    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[0][0] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        # initial conditions
        source = p

        for col in range(1, k):
            best_horizontal = 0
            best_horizontal_prob = -EPS
            for pair, prob in left_right_probs.items():
                if pair[0] == source and pair[1] in available and prob > best_horizontal_prob:
                    best_horizontal_prob = prob
                    best_horizontal = pair[1]
            source = best_horizontal
            labels[0][col] = best_horizontal
            available.remove(best_horizontal)

        source = p
        for row in range(1, k):
            best_vertical = 0
            best_vertical_prob = -EPS
            for pair, prob in up_down_probs.items():
                if pair[0] == source and pair[1] in available and prob > best_vertical_prob:
                    best_vertical_prob = prob
                    best_vertical = pair[1]
            source = best_vertical
            labels[row][0] = best_vertical
            available.remove(best_vertical)

        for i in range(1, k):
            for j in range(1, k):
                best_prob = -EPS
                best_piece = 0
                for piece in available:
                    prob = up_down_probs[(labels[i - 1][j], piece)] * left_right_probs[(labels[i][j - 1], piece)]
                    if prob >= best_prob:
                        best_prob = prob
                        best_piece = piece
                labels[i][j] = best_piece
                available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels

    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[0][0] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        # initial conditions
        source = p
        for row in range(1, k):
            best_vertical = 0
            best_vertical_prob = -EPS
            for pair, prob in up_down_probs.items():
                if pair[0] == source and pair[1] in available and prob > best_vertical_prob:
                    best_vertical_prob = prob
                    best_vertical = pair[1]
            source = best_vertical
            labels[row][0] = best_vertical
            available.remove(best_vertical)

        source = p

        for col in range(1, k):
            best_horizontal = 0
            best_horizontal_prob = -EPS
            for pair, prob in left_right_probs.items():
                if pair[0] == source and pair[1] in available and prob > best_horizontal_prob:
                    best_horizontal_prob = prob
                    best_horizontal = pair[1]
            source = best_horizontal
            labels[0][col] = best_horizontal
            available.remove(best_horizontal)

        for i in range(1, k):
            for j in range(1, k):
                best_prob = -EPS
                best_piece = 0
                for piece in available:
                    prob = up_down_probs[(labels[i - 1][j], piece)] * left_right_probs[(labels[i][j - 1], piece)]
                    if prob >= best_prob:
                        best_prob = prob
                        best_piece = piece
                labels[i][j] = best_piece
                available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels

    return list(best_labels.flatten()), best_log_prob


def greedy_bottom_left(left_right_probs, up_down_probs, k, images):
    best_labels = None
    best_log_prob = float('-inf')

    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[k - 1][0] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        # initial conditions
        source = p

        for col in range(1, k):
            best_horizontal = 0
            best_horizontal_prob = -EPS
            for pair, prob in left_right_probs.items():
                if pair[0] == source and pair[1] in available and prob > best_horizontal_prob:
                    best_horizontal_prob = prob
                    best_horizontal = pair[1]
            source = best_horizontal
            labels[k - 1][col] = best_horizontal
            available.remove(best_horizontal)

        source = p
        for row in range(k - 2, -1, -1):
            best_vertical = 0
            best_vertical_prob = -EPS
            for pair, prob in up_down_probs.items():
                # print(pair, prob)
                if pair[1] == source and pair[0] in available and prob > best_vertical_prob:
                    best_vertical_prob = prob
                    best_vertical = pair[0]
            source = best_vertical
            labels[row][0] = best_vertical
            available.remove(best_vertical)

        for i in range(k - 2, -1, -1):
            for j in range(1, k):
                best_prob = -EPS
                best_piece = 0
                for piece in available:
                    prob = up_down_probs[piece, (labels[i + 1][j])] * left_right_probs[(labels[i][j - 1], piece)]
                    if prob >= best_prob:
                        best_prob = prob
                        best_piece = piece
                labels[i][j] = best_piece
                available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels

    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[k - 1][0] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        # initial conditions
        source = p
        for row in range(k - 2, -1, -1):
            best_vertical = 0
            best_vertical_prob = -EPS
            for pair, prob in up_down_probs.items():
                if pair[1] == source and pair[0] in available and prob > best_vertical_prob:
                    best_vertical_prob = prob
                    best_vertical = pair[0]
            source = best_vertical
            labels[row][0] = best_vertical
            available.remove(best_vertical)

        source = p
        for col in range(1, k):
            best_horizontal = 0
            best_horizontal_prob = -EPS
            for pair, prob in left_right_probs.items():
                if pair[0] == source and pair[1] in available and prob > best_horizontal_prob:
                    best_horizontal_prob = prob
                    best_horizontal = pair[1]
            source = best_horizontal
            labels[k - 1][col] = best_horizontal
            available.remove(best_horizontal)

        for i in range(k - 2, -1, -1):
            for j in range(1, k):
                best_prob = -EPS
                best_piece = 0
                for piece in available:
                    prob = up_down_probs[piece, (labels[i + 1][j])] * left_right_probs[(labels[i][j - 1], piece)]
                    if prob >= best_prob:
                        best_prob = prob
                        best_piece = piece
                labels[i][j] = best_piece
                available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels

    return list(best_labels.flatten()), best_log_prob


def greedy_top_right(left_right_probs, up_down_probs, k, images):
    best_labels = None
    best_log_prob = float('-inf')

    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[0][k - 1] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        # initial conditions
        source = p

        for col in range(k - 2, -1, -1):
            best_horizontal = 0
            best_horizontal_prob = -EPS
            for pair, prob in left_right_probs.items():
                if pair[1] == source and pair[0] in available and prob > best_horizontal_prob:
                    best_horizontal_prob = prob
                    best_horizontal = pair[0]
            source = best_horizontal
            labels[0][col] = best_horizontal
            available.remove(best_horizontal)

        source = p
        for row in range(1, k):
            best_vertical = 0
            best_vertical_prob = -EPS
            for pair, prob in up_down_probs.items():
                if pair[0] == source and pair[1] in available and prob > best_vertical_prob:
                    best_vertical_prob = prob
                    best_vertical = pair[1]
            source = best_vertical
            labels[row][k - 1] = best_vertical
            available.remove(best_vertical)
        for i in range(1, k):
            for j in range(k - 2, -1, -1):
                best_prob = -EPS
                best_piece = 0
                for piece in available:
                    prob = up_down_probs[(labels[i - 1][j], piece)] * left_right_probs[piece, (labels[i][j + 1])]
                    if prob >= best_prob:
                        best_prob = prob
                        best_piece = piece
                labels[i][j] = best_piece
                available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels

    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[0][k - 1] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        # initial conditions

        source = p
        for row in range(1, k):
            best_vertical = 0
            best_vertical_prob = -EPS
            for pair, prob in up_down_probs.items():
                if pair[0] == source and pair[1] in available and prob > best_vertical_prob:
                    best_vertical_prob = prob
                    best_vertical = pair[1]
            source = best_vertical
            labels[row][k - 1] = best_vertical
            available.remove(best_vertical)

        source = p
        for col in range(k - 2, -1, -1):
            best_horizontal = 0
            best_horizontal_prob = -EPS
            for pair, prob in left_right_probs.items():
                if pair[1] == source and pair[0] in available and prob > best_horizontal_prob:
                    best_horizontal_prob = prob
                    best_horizontal = pair[0]
            source = best_horizontal
            labels[0][col] = best_horizontal
            available.remove(best_horizontal)

        for i in range(1, k):
            for j in range(k - 2, -1, -1):
                best_prob = -EPS
                best_piece = 0
                for piece in available:
                    prob = up_down_probs[(labels[i - 1][j], piece)] * left_right_probs[piece, (labels[i][j + 1])]
                    if prob >= best_prob:
                        best_prob = prob
                        best_piece = piece
                labels[i][j] = best_piece
                available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels
    return list(best_labels.flatten()), best_log_prob


def greedy_bottom_right(left_right_probs, up_down_probs, k, images):
    best_labels = None
    best_log_prob = float('-inf')

    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[k - 1][k - 1] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        # initial conditions
        source = p

        for col in range(k - 2, -1, -1):
            best_horizontal = 0
            best_horizontal_prob = -EPS
            for pair, prob in left_right_probs.items():
                if pair[1] == source and pair[0] in available and prob > best_horizontal_prob:
                    best_horizontal_prob = prob
                    best_horizontal = pair[0]
            source = best_horizontal
            labels[k - 1][col] = best_horizontal
            available.remove(best_horizontal)

        source = p
        for row in range(k - 2, -1, -1):
            best_vertical = 0
            best_vertical_prob = -EPS
            for pair, prob in up_down_probs.items():
                if pair[1] == source and pair[0] in available and prob > best_vertical_prob:
                    best_vertical_prob = prob
                    best_vertical = pair[0]
            source = best_vertical
            labels[row][k - 1] = best_vertical
            available.remove(best_vertical)
        for i in range(k - 2, -1, -1):
            for j in range(k - 2, -1, -1):
                best_prob = -EPS
                best_piece = 0
                for piece in available:
                    prob = up_down_probs[piece, (labels[i + 1][j])] * left_right_probs[piece, (labels[i][j + 1])]
                    if prob >= best_prob:
                        best_prob = prob
                        best_piece = piece
                labels[i][j] = best_piece
                available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels

    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[k - 1][k - 1] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        # initial conditions

        source = p
        for row in range(k - 2, -1, -1):
            best_vertical = 0
            best_vertical_prob = -EPS
            for pair, prob in up_down_probs.items():
                if pair[1] == source and pair[0] in available and prob > best_vertical_prob:
                    best_vertical_prob = prob
                    best_vertical = pair[0]
            source = best_vertical
            labels[row][k - 1] = best_vertical
            available.remove(best_vertical)

        source = p
        for col in range(k - 2, -1, -1):
            best_horizontal = 0
            best_horizontal_prob = -EPS
            for pair, prob in left_right_probs.items():
                if pair[1] == source and pair[0] in available and prob > best_horizontal_prob:
                    best_horizontal_prob = prob
                    best_horizontal = pair[0]
            source = best_horizontal
            labels[k - 1][col] = best_horizontal
            available.remove(best_horizontal)

        for i in range(k - 2, -1, -1):
            for j in range(k - 2, -1, -1):
                best_prob = -EPS
                best_piece = 0
                for piece in available:
                    prob = up_down_probs[piece, (labels[i + 1][j])] * left_right_probs[piece, (labels[i][j + 1])]
                    if prob >= best_prob:
                        best_prob = prob
                        best_piece = piece
                labels[i][j] = best_piece
                available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels

    return list(best_labels.flatten()), best_log_prob


def all_permutes(left_right_probs, up_down_probs, k):
    best_labels = None
    best_log_prob = float('-inf')

    for labels in permutations(range(1, k * k + 1)):
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels

    return best_labels, best_log_prob


def get_next_index_1(row, column):
    if column > row:
        return column, row
    if row == column:
        return 0, column + 1
    if row - column == 1:
        return row, column + 1
    if row - column > 1:
        return column + 1, row


def greedy_diagonal_1(left_right_probs, up_down_probs, k):
    row, col = 0, 0
    best_log_prob = float('-inf')
    best_labels = None
    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[0][0] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        while bool(available) is True:
            row, col = get_next_index_1(row, col)
            best_prob = -EPS
            best_piece = 0
            for piece in available:
                if row == 0:
                    prob = left_right_probs[(labels[row][col - 1]), piece]
                if col == 0:
                    prob = up_down_probs[(labels[row - 1][col]), piece]
                if row != 0 and col != 0:
                    prob = up_down_probs[(labels[row - 1][col]), piece] * left_right_probs[
                        (labels[row][col - 1]), piece]
                if prob >= best_prob:
                    best_prob = prob
                    best_piece = piece
            labels[row][col] = best_piece
            available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        row, col = 0, 0
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels
    return best_labels, best_log_prob


def get_next_index_2(row, column, k):
    row = k - 1 - row
    if column > row:
        return k - 1 - column, row
    if row == column:
        return k - 1, column + 1
    if row - column == 1:
        return k - 1 - row, column + 1
    if row - column > 1:
        return k - 1 - column - 1, row


def greedy_diagonal_2(left_right_probs, up_down_probs, k):
    row, col = k - 1, 0
    best_log_prob = float('-inf')
    best_labels = None
    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[k - 1][0] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        while bool(available) is True:
            row, col = get_next_index_2(row, col, k)
            best_prob = -EPS
            best_piece = 0
            for piece in available:
                if row == k - 1:
                    prob = left_right_probs[labels[row][col - 1], piece]
                if col == 0:
                    prob = up_down_probs[piece, labels[row + 1][col]]
                if row != k - 1 and col != 0:
                    prob = up_down_probs[piece, labels[row + 1][col]] * left_right_probs[labels[row][col - 1], piece]
                if prob >= best_prob:
                    best_prob = prob
                    best_piece = piece
            labels[row][col] = best_piece
            available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        row, col = k - 1, 0
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels
    return best_labels, best_log_prob


def get_next_index_3(row, column, k):
    column = k - 1 - column
    if row > column:
        return column, k - 1 - row
    if row == column:
        return row + 1, k - 1
    if column - row == 1:
        return row + 1, k - 1 - column
    if column - row > 1:
        return column, k - 1 - row - 1


def greedy_diagonal_3(left_right_probs, up_down_probs, k):
    row, col = 0, k - 1
    best_log_prob = float('-inf')
    best_labels = None
    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[0][k - 1] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        while bool(available) is True:
            row, col = get_next_index_3(row, col, k)
            best_prob = -EPS
            best_piece = 0
            for piece in available:
                if row == 0:
                    prob = left_right_probs[piece, labels[row][col + 1]]
                if col == k - 1:
                    prob = up_down_probs[labels[row - 1][col], piece]
                if row != 0 and col != k - 1:
                    prob = up_down_probs[labels[row - 1][col], piece] * left_right_probs[piece, labels[row][col + 1]]
                if prob >= best_prob:
                    best_prob = prob
                    best_piece = piece
            labels[row][col] = best_piece
            available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        row, col = 0, k - 1
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels
    return best_labels, best_log_prob


def get_next_index_4(row, column, k):
    column = k - 1 - column
    row = k - 1 - row
    if row > column:
        return k - 1 - column, k - 1 - row
    if row == column:
        return k - 1 - row - 1, k - 1
    if column - row == 1:
        return k - 1 - row - 1, k - 1 - column
    if column - row > 1:
        return k - 1 - column, k - 1 - row - 1


def greedy_diagonal_4(left_right_probs, up_down_probs, k):
    row, col = k - 1, k - 1
    best_log_prob = float('-inf')
    best_labels = None
    for p in range(1, k * k + 1):
        labels = np.zeros((k, k), dtype=np.int32)
        labels[k - 1][k - 1] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        while bool(available) is True:
            row, col = get_next_index_4(row, col, k)
            best_prob = -EPS
            best_piece = 0
            for piece in available:
                if row == k - 1:
                    prob = left_right_probs[piece, labels[row][col + 1]]
                if col == k - 1:
                    prob = up_down_probs[piece, labels[row + 1][col]]
                if row != k - 1 and col != k - 1:
                    prob = up_down_probs[piece, labels[row + 1][col]] * left_right_probs[piece, labels[row][col + 1]]
                if prob >= best_prob:
                    best_prob = prob
                    best_piece = piece
            labels[row][col] = best_piece
            available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        row, col = k - 1, k - 1
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels
    return best_labels, best_log_prob


def infer(left_right_probs, up_down_probs, k, images):
    if k == 2:
        return all_permutes(left_right_probs, up_down_probs, k)
    labels = list()
    log_prob = list()

    # probability_estimator(left_right_probs, up_down_probs, k)
    top_left_labels, top_left_log_prob = greedy_top_left(left_right_probs, up_down_probs, k, images)
    labels.append(top_left_labels)
    log_prob.append(top_left_log_prob)
    bottom_left_labels, bottom_left_log_prob = greedy_bottom_left(left_right_probs, up_down_probs, k, images)
    labels.append(bottom_left_labels)
    log_prob.append(bottom_left_log_prob)
    top_right_labels, top_right_log_prob = greedy_top_right(left_right_probs, up_down_probs, k, images)
    labels.append(top_right_labels)
    log_prob.append(top_right_log_prob)
    bottom_right_labels, bottom_right_log_prob = greedy_bottom_right(left_right_probs, up_down_probs, k, images)
    labels.append(bottom_right_labels)
    log_prob.append(bottom_right_log_prob)

    best_diagonal_labels, best_diagonal_log_prob = infer2(left_right_probs, up_down_probs, k, images)
    labels.append(best_diagonal_labels)
    log_prob.append(best_diagonal_log_prob)

    center_labels, center_log_prob = greedy_center(left_right_probs, up_down_probs, k)
    labels.append(center_labels)
    log_prob.append(center_log_prob)

    # union_labels, union_log_prob = greedy_doc(left_right_probs, up_down_probs, k)
    # if union_log_prob is not None:
    #     labels.append(union_labels)
    #     log_prob.append(union_log_prob)
    # else:
    #     print('union failed')

    best_log_prob = float('-inf')
    for prob, label in zip(log_prob, labels):
        if prob > best_log_prob:
            best_log_prob = prob
            best_labels = label
    return best_labels, best_log_prob


def infer2(left_right_probs, up_down_probs, k, images):
    labels = list()
    log_prob = list()
    best_labels_1, best_log_prob_1 = greedy_diagonal_1(left_right_probs, up_down_probs, k)
    best_labels_2, best_log_prob_2 = greedy_diagonal_2(left_right_probs, up_down_probs, k)
    best_labels_3, best_log_prob_3 = greedy_diagonal_3(left_right_probs, up_down_probs, k)
    best_labels_4, best_log_prob_4 = greedy_diagonal_4(left_right_probs, up_down_probs, k)
    labels.append(best_labels_1)
    log_prob.append(best_log_prob_1)
    labels.append(best_labels_2)
    log_prob.append(best_log_prob_2)
    labels.append(best_labels_3)
    log_prob.append(best_log_prob_3)
    labels.append(best_labels_4)
    log_prob.append(best_log_prob_4)

    best_log_prob = float('-inf')
    for prob, label in zip(log_prob, labels):
        if prob > best_log_prob:
            best_log_prob = prob
            best_labels = label
    return list(best_labels.flatten()), best_log_prob


def greedy_doc(left_right_probs, up_down_probs, k):
    sorted_lr = sorted(left_right_probs.items(), key=operator.itemgetter(1), reverse=True)
    object_array = np.zeros((k * k, 2))
    rows = dict()
    available = k * k
    next_row = 1
    for idx in range(0, k * k):
        object_array[idx][0] = idx + 1
    for pair, prob in sorted_lr:
        if available == 0 and len(rows) == k:
            break
        row_1 = object_array[pair[0] - 1][1]
        row_2 = object_array[pair[1] - 1][1]
        if row_1 == row_2 != 0:  # numbers belong to same row
            continue
        if row_1 == row_2 == 0:  # numbers don't belong to rows
            new_row = next_row
            next_row += 1
            rows[new_row] = list(pair)
            object_array[pair[0] - 1][1] = new_row
            object_array[pair[1] - 1][1] = new_row
            available -= 2
            continue
        if row_1 == 0:
            if len(rows[row_2]) == k:  # row already full
                continue
            if rows[row_2][0] != pair[1]:  # pair[1] already left neighbor
                continue
            rows[row_2].insert(0, pair[0])
            object_array[pair[0] - 1][1] = row_2
            available -= 1
            continue
        if row_2 == 0:
            if len(rows[row_1]) == k:  # row already full
                continue
            if rows[row_1][-1] != pair[0]:  # pair[0] already right neighbor
                continue
            rows[row_1].append(pair[1])
            object_array[pair[1] - 1][1] = row_1
            available -= 1
            continue
        if row_1 != row_2 != 0:
            if len(rows[row_1]) + len(rows[row_2]) > k:  # union is bigger than k
                continue
            if rows[row_1][-1] != pair[0] or rows[row_2][0] != pair[1]:
                continue
            rows[row_1].extend(rows.pop(row_2))
            for num in rows[row_1]:
                object_array[num - 1][1] = row_1
    if len(rows) == k:
        labels = np.zeros((k, k), dtype=np.int32)
        for num, row in enumerate(rows.values()):
            labels[num] = row
        return all_permute_doc(labels, left_right_probs, up_down_probs, k)
    else:
        return None, None


def all_permute_doc(labels, left_right_probs, up_down_probs, k):
    best_log_prob = float('-inf')
    best_labels = None
    row_permutations = list(permutations(range(0, k)))
    for perm in row_permutations:
        labels_temp = labels[perm, :]
        log_prob = get_inference_log_prob(labels_temp, left_right_probs, up_down_probs, k)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels_temp
    return best_labels, best_log_prob


def get_next_index_center(row, column, k):
    k_4_lookup_table = np.array([[(3, 1), (0, 2), (0, 3), (0, 0)],
                                 [(2, 0), (1, 2), (2, 1), (2, 3)],
                                 [(0, 1), (2, 2), (1, 3), (1, 0)],
                                 [(None, None), (3, 2), (3, 3), (3, 0)]])
    k_5_lookup_table = np.array([[(4, 2), (0, 4), (0, 3), (0, 1), (0, 0)],
                                 [(0, 2), (1, 3), (1, 1), (2, 4), (2, 0)],
                                 [(3, 0), (3, 2), (2, 3), (2, 1), (3, 4)],
                                 [(1, 0), (1, 2), (3, 3), (3, 1), (1, 4)],
                                 [(None, None), (4, 4), (4, 3), (4, 1), (4, 0)]])
    if k == 4:
        return k_4_lookup_table[row][column]
    if k == 5:
        return k_5_lookup_table[row][column]


def greedy_center(left_right_probs, up_down_probs, k):
    best_log_prob = float('-inf')
    best_labels = None
    for p in range(1, k * k + 1):
        if k == 4:
            row, col = 1, 1
        else:  # k == 5:
            row, col = 2, 2
        labels = np.zeros((k, k), dtype=np.int32)
        labels[row][col] = p
        available = set(range(1, k * k + 1))
        available.remove(p)
        while bool(available) is True:
            row, col = get_next_index_center(row, col, k)
            best_prob = -EPS
            best_piece = 0
            for piece in available:
                if col + 1 < k and labels[row][col + 1] != 0:
                    right_prob = left_right_probs[piece, labels[row][col + 1]]
                else:
                    right_prob = 1
                if col - 1 > -1 and labels[row][col - 1] != 0:
                    left_prob = left_right_probs[labels[row][col - 1], piece]
                else:
                    left_prob = 1
                if row + 1 < k and labels[row + 1][col] != 0:
                    down_prob = up_down_probs[piece, labels[row + 1][col]]
                else:
                    down_prob = 1
                if row - 1 > -1 and labels[row - 1][col] != 0:
                    up_prob = up_down_probs[labels[row - 1][col], piece]
                else:
                    up_prob = 1
                prob = right_prob * left_prob * up_prob * down_prob
                if prob >= best_prob:
                    best_prob = prob
                    best_piece = piece
            labels[row][col] = best_piece
            available.remove(best_piece)
        log_prob = get_inference_log_prob(labels, left_right_probs, up_down_probs, k)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_labels = labels
    return list(best_labels.flatten()), best_log_prob


def dist(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


def probability_estimator(left_right_probs, up_down_probs, k):
    available = set(range(1, k * k + 1))
    for piece in available:
        neighbors = dict()
        sum = 0
        for k, v in left_right_probs.items():
            if k[0] != piece:
                continue
            neighbors[k[1]] = v
            sum += v
        for k, prob in neighbors.items():
            left_right_probs[piece, k] = prob / sum
        neighbors = dict()
        sum = 0
        for k, v in up_down_probs.items():
            if k[0] != piece:
                continue
            neighbors[k[1]] = v
            sum += v
        for k, prob in neighbors.items():
            up_down_probs[piece, k] = prob / sum


def infer_by_pos(pred_positions, k):
    weights = np.zeros((k ** 2, k ** 2))
    actual_positions = utils.tile_rel_center[k]

    for i in range(k ** 2):
        for j in range(k ** 2):
            weights[i][j] = dist(pred_positions[i], actual_positions[j])

    row_ind, col_ind = linear_sum_assignment(weights)

    labels = np.zeros((k ** 2))
    labels[row_ind] = col_ind

    return labels


class Solver:

    def __init__(self, is_regression=False):
        self.is_regression = is_regression
        self.doc_image_classifier = DocImageClassifier(load_model=True)
        self.doc_lr_neighbors_classifier = NeighborsClassifier(is_docs=True, is_left_right=True, load_model=True)
        self.doc_ud_neighbors_classifier = NeighborsClassifier(is_docs=True, is_left_right=False, load_model=True)
        self.img_lr_neighbors_classifier = NeighborsClassifier(is_docs=False, is_left_right=True, load_model=True)
        self.img_ud_neighbors_classifier = NeighborsClassifier(is_docs=False, is_left_right=False, load_model=True)

        self.tile_pos_regression = TilePosRegressor(load_model=True)

    def predict_neigbors(self, images, is_img):
        left_right_probs = dict()
        up_down_probs = dict()
        k = {4: 2, 16: 4, 25: 5}[len(images)]

        # is_doc = np.round(self.doc_image_classifier.predict(images_for_classification)) == utils.DOCUMENT
        is_doc = not is_img
        lr_classifier = self.doc_lr_neighbors_classifier if is_doc else self.img_lr_neighbors_classifier
        ud_classifier = self.doc_ud_neighbors_classifier if is_doc else self.img_ud_neighbors_classifier

        for i1, img1 in enumerate(images, 1):
            for i2, img2 in enumerate(images, 1):
                if i1 != i2:
                    left_right_probs[(i1, i2)] = lr_classifier.predict(img1, img2, strip_size=10)[0][0]
                    up_down_probs[(i1, i2)] = ud_classifier.predict(img1, img2, strip_size=10)[0][0]

        return infer(left_right_probs, up_down_probs, k, images)

    def predict_regression(self, images):
        predictions = self.tile_pos_regression.predict(images)
        k = {4: 2, 16: 4, 25: 5}[len(images)]

        predictions = predictions  # preprocess

        return infer_by_pos(predictions, k)

    def predict(self, images, is_img):
        if self.is_regression:
            return self.predict_regression(images)
        else:
            return self.predict_neigbors(images, is_img)

    def evaluate(self, images, labels, is_img):

        predictions, log_prob = self.predict(images, is_img)
        acc = np.mean(np.asarray(predictions) == np.asarray(labels))

        return acc, predictions, log_prob


if __name__ == '__main__':
    X_img_train, X_img_test, X_img_val = utils.split_train_test_val(utils.DOC_PATH, 0.7, 0.15, seed=42)
    solver = Solver()

    error_prob = utils.AverageMeter()
    corre_prob = utils.AverageMeter()
    error_counter = 0
    sum = 0
    count = 0
    error_4_count = 0
    sum_4 = utils.AverageMeter()
    sum_16 = utils.AverageMeter()
    sum_25 = utils.AverageMeter()
    for idx, path in enumerate(X_img_test):
        images = list(utils.get_images_from_path(path))
        tiles = len(images)
        count += tiles
        labels = list(range(1, tiles + 1))
        images, labels = shuffle(images, labels)
        new_labels = list(range(1, tiles + 1))

        # process image & predict if it's an image or doc
        images = utils.process_images(images)
        is_image = np.round(solver.doc_image_classifier.predict(images)) == utils.IMAGE
        images = utils.normalize_images(images, is_img=is_image)
        for i, label in enumerate(labels, 1):
            new_labels[label - 1] = i
        labels = new_labels
        acc, predictions, log_prob = solver.evaluate(images, labels, is_image)
        if acc < 1:
            error_counter += 1
            error_prob.update(log_prob)
        else:
            corre_prob.update(log_prob)
        sum += acc * tiles
        if tiles == 4:
            sum_4.update(acc)
        if tiles == 16:
            sum_16.update(acc)
        if tiles == 25:
            sum_25.update(acc)
        # print('correct probability [{:.3f}], error probability [{:.3f}]'.format(corre_prob.avg, error_prob.avg))
        print('{}\t {} tiles, {} acc\t| 4 tiles - [{:.3f}], 16 tiles - [{:.3f}], 25 tiles - [{:.3f}]'
              .format(path, tiles, acc, sum_4.avg, sum_16.avg, sum_25.avg))

    print("Overall accuracy: %f" % (sum / count))
    print('misclassified {} samples of {} total samples'.format(error_counter, idx))
