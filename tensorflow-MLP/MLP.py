# -*- coding: utf-8 -*-
"""
    standard multi-layer perceptron
"""
from Layer import *
import tensorflow as tf
import numpy
import os
import sys


class MLP(object):
    def __init__(self, layer_sizes, layer_types,
                 init_value_scale=1.0, uniform_init=False,
                 verbose = True):
        '''
        initialize network architecture
        :param layer_sizes: list type, layer sizes,
                e.g. a 3-layer network "784:256:10"
        :param layer_types: list type, hidden layer types,
                e.g. sigmoid/tanh or "sigmoid:tanh" for 2-hidden-layer network
        :param init_value_scale: int, scale for uniform initialization
        :param uniform_init: bool, true for uniform, gaussian otherwise
        :param verbose: bool, verbose
        :return:
        '''

        self.verbose = verbose
        # input settings
        self.x = tf.placeholder(tf.float32, [None, layer_sizes[0]], name='input')
        self.y = tf.placeholder(tf.float32, [None, layer_sizes[-1]], name='truth')
        self.learning_rate = tf.placeholder(tf.float32, name='learningrate')
        self.momentum = tf.placeholder(tf.float32, name='momentum')
        # layers
        self.layers = []
        # build multi-layer perceptron architecture
        if self.verbose: print('Building Multilayer Perceptron...')
        # forward pass and build output
        for idx in xrange(len(layer_sizes) - 1):
            n_input = layer_sizes[idx]
            n_output = layer_sizes[idx + 1]
            layer = Layer(n_input, n_output, layer_types[idx], init_value_scale, uniform_init)
            self.layers.append(layer)

        # forward
        net_output = self.x
        for idx in xrange(len(self.layers)):
            net_output = self.layers[idx].output(net_output)
        # cost function with ground truth provided, for training
        self.cost = self.layers[-1].neg_loglikelihood(net_output, self.y)
        # make prediction
        self.prediction = tf.arg_max(net_output, dimension=1)
        # prediction error
        self.prederr = tf.reduce_mean(tf.to_float(tf.not_equal(self.prediction, tf.arg_max(self.y, dimension=1))))
        # training
        self.train_process = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)
        # session
        self.sess = tf.Session()

    def train(self, x, y, epoch_size=0,
              max_epoch=20, minibatch_size=25,
              learning_rate=0.1, momentum=0.9):
        '''
        network training
        :param x: numpy.ndarray
            training data of shape (n_instances, n_features)
        :param y: numpy.ndarray
            labels of training data of shape (n_instance,)
        :param epoch_size: epochSize=0 means epochSize is the size of
            the training set. Must be evenly divisible
            into number of data frames.
        :param max_epoch: maximum epochs
        :param minibatch_size: mini batch size
        :param learning_rate: learning rate
        :param momentum: momentum
        :return:
        '''
        if epoch_size == 0:
            epoch_size = x.shape[0]
        assert x.shape[0] % minibatch_size == 0
        if self.verbose: print 'Start training MLP'
        random_ins_index = numpy.arange(x.shape[0])
        self.sess.run(tf.initialize_all_variables())
        for epoch_idx in xrange(max_epoch):
            numpy.random.shuffle(random_ins_index) # shuffle instance indexs
            ins_start_idx = 0
            while ins_start_idx < epoch_size:
                self.sess.run(self.train_process, feed_dict={
                    self.x: x[random_ins_index[ins_start_idx:ins_start_idx+minibatch_size],:],
                    self.y: y[random_ins_index[ins_start_idx:ins_start_idx+minibatch_size]],
                    self.learning_rate: learning_rate,
                    self.momentum: momentum
                })
                ins_start_idx += minibatch_size
            if self.verbose:
                cost = self.sess.run(self.cost, feed_dict={
                    self.x: x, self.y: y
                })
                prederr = self.sess.run(self.prederr, feed_dict={
                    self.x: x, self.y: y
                })

                print 'in epoch %d: cost: %f, prediction error: %f' % (epoch_idx, cost, prederr)
        if self.verbose: print 'Finish training. Training error: %f' % prederr

    def test(self, x, y):
        '''
        test network
        :param x: numpy.ndarray
            test data of shape (n_instances, n_features)
        :param y: numpy.ndarray
            ground truth
        :return: prediction error
        '''
        return self.sess.run(self.prederr, feed_dict={
            self.x: x, self.y: y
        })

    def predict(self, x):
        '''
        make predictions
        :param x: numpy.ndarray
            test data of shape (n_instances, n_features)
        :return: numpy.ndarray
            predictions
        '''
        return self.sess.run(self.prediction, feed_dict={
            self.x: x
        })

