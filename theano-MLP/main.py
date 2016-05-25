# -*- coding: utf-8 -*-
import sys
from sklearn.datasets import fetch_mldata
from MLP import *


if __name__ == "__main__":
    # fetch data from sklearn
    mnist = fetch_mldata('MNIST original')
    train_data = mnist.data[0:60000, :] / 255. # normalize
    train_label = mnist.target[0:60000].astype(int)
    test_data = mnist.data[60000:70000, :] / 255. # normalize
    test_label = mnist.target[60000:70000].astype(int)

    # construct network
    mlp = MLP(layer_sizes=[train_data.shape[1], 256, 10],
              layer_types=['sigmoid', 'softmax'],
              uniform_init=False)
    mlp.train(train_data, train_label)
    print 'Test error: %f' % mlp.test(test_data, test_label)
