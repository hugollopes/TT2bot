from __future__ import print_function
import numpy as np


def make_arrays(nb_rows, img_size_x,img_size_y):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size_x, img_size_y), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
