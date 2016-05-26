# -*- coding: utf-8 -*-
import numpy
import tensorflow as tf


class Layer:
    def __init__(self, n_input, n_output, activation = None,
                 init_value_scale=1.0, uniform_init=False):
        '''
        A layer of a neural network, computes s(xW + b) where s is a nonlinearity and x is the input vector.
        :param n_input: input dimension
        :param n_output: output dimension
        :param activation: string or an elemwise function
                Activation function for layer output
        :param init_value_scale: uniformly initialization parameter,
                [-0.05*init_value_scale, 0.05*init_value_scale]
        :param uniform_init: True if use uniform distribution to the network parameters,
                or normal distribution, otherwise
        :return:
        '''
        # weight
        self.W = tf.Variable(tf.random_uniform([n_input, n_output],
                                               minval=-0.05*numpy.abs(init_value_scale),
                                               maxval=0.05*numpy.abs(init_value_scale))) \
            if uniform_init else tf.Variable(tf.random_normal([n_input, n_output]))
        # bias
        self.b = tf.Variable(tf.random_uniform([n_output],
                                               minval=-0.05*numpy.abs(init_value_scale),
                                               maxval=0.05*numpy.abs(init_value_scale))) \
            if uniform_init else tf.Variable(tf.random_normal([n_output]))
        self.activation = get_act(activation) if type(activation) is str else activation

    def output(self, x):
        '''
        Compute this layer's output given an input
        :param x: Tensorflow.placeholder
                Tensorflow symbolic variable for layer input
        :returns:
        '''
        # compute linear mix
        lin_output = tf.matmul(x, self.W) + self.b
        # Output is just linear mix if no activation function
        # Otherwise, apply the actiovation function
        return (lin_output if self.activation is None else self.activation(lin_output))

    def neg_loglikelihood(self, x, y):
        '''
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        :param x: Tensorflow.Variable
                Tensorflow symbolic variable for layer input
        :param y: Tensorflow.placeholder
                corresponds to a vector that gives for each example the correct label;
        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        '''
        return -tf.reduce_mean(tf.reduce_sum(tf.log(x) * y, reduction_indices=1))


# get activation function by string
def get_act(activation = None):
    if activation is None:
        return None
    activation = activation.lower()
    if activation == 'sigmoid':
        return tf.nn.sigmoid
    if activation == 'tanh':
        return tf.nn.tanh
    if activation == 'relu':
        return tf.nn.relu
    if activation == 'softmax':
        return tf.nn.softmax
    print ('Unrecognized active function. layerType: %s' % activation)
    sys.exit()