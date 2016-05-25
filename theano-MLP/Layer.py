# -*- coding: utf-8 -*-
import numpy as np
import theano
from theano import tensor as T



class Layer:
    def __init__(self, W_init, b_init, activation = None):
        '''
        A layer of a neural network, computes s(xW + b) where s is a nonlinearity and x is the input vector.
        :param W_init: np.ndarray, shape=(n_input, n_output)
                Values to initialize the weight matrix to.
        :param b_init: np.ndarray, shape=(n_output,)
                Values to initialize the bias vector
        :param activation: string or an elemwise function
                Activation function for layer output
        :return:
        '''
        # Retrieve the input and output dimensionality based on W's initialization
        n_input, n_output = W_init.shape if type(W_init) is np.ndarray else W_init.get_value().shape
        # Make sure b is n_output in size
        assert b_init.shape == (n_output, )
        # All parameters should be shared variables.
        # They're used in this class to compute the layer output
        # but are updated elsewhere when optimizing the network parameters.
        # Note that we are explicitly requiring that W_init has the theano.config.floatX dtype
        self.W = theano.shared(value = W_init.astype(theano.config.floatX) \
                               if type(W_init) is np.ndarray else W_init.get_value(),
                               # The name parameter is solely for printing purposes
                               name = 'W')
        # We can force our bais vector b to be a column vector using numpy's reshape method
        # When b is a column vector, we can pass a matrix-shaped input to the layer
        # and get a matrix-shaped output, thanks to broadcasting
        self.b = theano.shared(value = b_init.astype(theano.config.floatX) \
                               if type(b_init) is np.ndarray else b_init.get_value(),
                               name = 'b')
        self.activation = get_act(activation) if type(activation) is str else activation
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list
        self.params = [self.W, self.b]

    def output(self, x):
        '''
        Compute this layer's output given an input
        :param x: theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input
        :returns: theano.tensor.var.TensorVariable
                Mixed, biased, and activated x
        '''
        # compute linear mix
        lin_output = T.dot(x, self.W) + self.b
        # Output is just linear mix if no activation function
        # Otherwise, apply the actiovation function
        return (lin_output if self.activation is None else self.activation(lin_output))

    def neg_loglikelihood(self, x, y):
        '''
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.
        :param y: corresponds to a vector that gives for each example the
                  correct label;
        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        '''
        return -T.mean(T.log(x)[T.arange(y.shape[0]), y])


def relu(x):
    return T.maximum(0, x)


# get activation function by string
def get_act(activation = None):
    if activation is None:
        return None
    activation = activation.lower()
    if activation == 'sigmoid':
        return T.nnet.sigmoid
    if activation == 'tanh':
        return T.tanh
    if activation == 'relu':
        return relu
    if activation == 'softmax':
        return T.nnet.softmax
    print ('Unrecognized active function. layerType: %s' % activation)
    sys.exit()