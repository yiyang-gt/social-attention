# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec
from keras.layers import *

import theano
from theano import tensor as T


class Summation(Layer):
    """
    Simply sum embeddings of all tokens of a sentence
    """
    def __init__(self, **kwargs):
        super(Summation, self).__init__(**kwargs)

    def call(self, x, mask=None):
        sums = x.sum(axis=1)
        return sums

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])


class Mixture(Layer):
    """
    Mixture model
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Mixture, self).__init__(**kwargs)

    def call(self, x, mask=None):
        """
        ensemble_layer: n_batch x n_model x output_dim
        densities: n_batch x n_model
        """
        ensemble_layer, densities = x[:,:,:-1], x[:,:,-1]
        return K.batch_dot(densities, ensemble_layer, [[1], [1]])

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

