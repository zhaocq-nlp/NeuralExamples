# -*- coding: utf-8 -*-
import sys
from sklearn.datasets import fetch_mldata
from MLP import *


def dense_to_one_hot(labels_dense, num_classes=10):
    """
    Convert class labels from scalars to one-hot vectors
    :param labels_dense:
    :param num_classes:
    :return:
    """
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


if __name__ == "__main__":
    # fetch data from sklearn
    mnist = fetch_mldata('MNIST original')
    train_data = mnist.data[0:60000, :] / 255. # normalize
    train_label = mnist.target[0:60000].astype(dtype=int)
    train_label_one_hot = dense_to_one_hot(train_label).astype(dtype=float)
    test_data = mnist.data[60000:70000, :] / 255. # normalize
    test_label = mnist.target[60000:70000].astype(dtype=int)
    test_label_one_hot = dense_to_one_hot(test_label).astype(dtype=float)

    # construct network
    mlp = MLP(layer_sizes=[train_data.shape[1], 256, 10],
              layer_types=['sigmoid', 'softmax'],
              uniform_init=False)
    mlp.train(train_data, train_label_one_hot)

    print 'Test error: %f' % mlp.test(test_data, test_label_one_hot)


