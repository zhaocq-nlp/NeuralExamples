# -*- coding: utf-8 -*-
"""
    standard multi-layer perceptron
"""
from Layer import *
from theano import tensor as T
import theano
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
        self.input = T.matrix(name = 'input', dtype = theano.config.floatX)
        self.truth = T.lvector(name = 'label')
        self.learning_rate = T.scalar(name = 'learn rate')
        self.momentum = T.scalar(name = 'momentum')
        # layers
        self.layers = []
        # all parameters in the model
        self.params = []
        # build multi-layer perceptron architecture
        if self.verbose: print('Building Multilayer Perceptron...')
        # forward pass and build output
        for idx in xrange(len(layer_sizes) - 1):
            n_input = layer_sizes[idx]
            n_output = layer_sizes[idx + 1]
            w = (numpy.random.random((n_input, n_output)) - 0.5) * init_value_scale * 0.05 \
                if uniform_init else numpy.random.randn(n_input, n_output)
            b = (numpy.random.random((n_output, )) - 0.5) * init_value_scale * 0.05 \
                if uniform_init else numpy.random.randn(n_output)
            layer = Layer(w, b, layer_types[idx])
            self.layers.append(layer)
            self.params.extend(layer.params)

        # forward
        net_output = self.input
        for idx in xrange(len(self.layers)):
            net_output = self.layers[idx].output(net_output)
        # cost function with ground truth provided, for training
        self.cost = self.layers[-1].neg_loglikelihood(net_output, self.truth)
        # make prediction
        self.predict = T.argmax(net_output, axis=1)

        # stochastic gradient descent
        self.updates = []
        for param in self.params:
            # For each parameter, we'll create a param_update shared variable.
            # This variable will keep track of the parameter's update step across iterations.
            # We initialize it to 0
            param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
            # Each parameter is updated by taking a step in the direction of the gradient.
            # However, we also "mix in" the previous step according to the given momentum value.
            # Note that when updating param_update, we are using its old value and also the new gradient step.
            self.updates.append((param, param - self.learning_rate * param_update))
            # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
            self.updates.append((param_update, self.momentum*param_update + (1. - self.momentum)*T.grad(self.cost, param)))

        # build functions
        # build objective function
        self.fobjective = theano.function(inputs = [self.input, self.truth, self.learning_rate, self.momentum],
                                        outputs = self.cost, on_unused_input='ignore', updates = self.updates)
        # build prediction function
        self.fpredict = theano.function(inputs = [self.input],
                                        outputs = self.predict)
        # build prediction error function
        self.fprederr = theano.function(inputs = [self.input, self.truth],
                                        outputs = (self.cost, T.mean(T.neq(self.predict, self.truth))))

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
        for epoch_idx in xrange(max_epoch):
            numpy.random.shuffle(random_ins_index) # shuffle instance indexs
            ins_start_idx = 0
            while ins_start_idx < epoch_size:
                self.fobjective(
                    x[random_ins_index[ins_start_idx:ins_start_idx+minibatch_size],:],
                    y[random_ins_index[ins_start_idx:ins_start_idx+minibatch_size]],
                    learning_rate, momentum)
                ins_start_idx += minibatch_size
            if self.verbose:
                cost, prederr = self.fprederr(x, y)
                print 'in epoch %d: cost: %f, prediction error: %f' % (epoch_idx, cost, prederr)
        if self.verbose: print 'Finish training. Training error: %f' % prederr

    def test(self, x, y):
        return self.fprederr(x, y)[1]

    def predict(self, x):
        '''
        make predictions
        :param x: numpy.ndarray
            test data of shape (n_instances, n_features)
        :return: numpy.ndarray
            predictions
        '''
        return self.fpredict(x)