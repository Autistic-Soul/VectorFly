#!/usr/bin/env Python
# -*- coding: utf-8 -*-

import cmath

class nerual(object):

    """
    weights:
    ===
        <list>: <int> or <float> or <complex>

    bias:
    ===
        <int> or <float> or <complex>

    activation_function:
    ===
        <str>: "SIGMOID" or "TANH" or "RELU"
    """

    def __init__( self, weights = None, bias = None, activation_function = None ):
        self.__W = weights
        self.__B = bias
        self.__fun = activation_function

    def Sigmoid(self, x):
        return 1 / ( 1 + cmath.exp(-x) )

    def tanh(self, x):
        return ( cmath.exp(x) - cmath.exp(-x) ) / ( cmath.exp(x) + cmath.exp(-x) )

    def th(self, x):
        return self.tanh(x)

    def ReLU(self, x):
        return max(0, x)

    def ComputeY( self, X = None ):

        y = sum([ self.__W[i] * X[i] for i in range(len(self.__W)) ]) + self.__B

        if self.__fun == "SIGMOID":
            y = self.Sigmoid(y)
        elif self.__fun == "TANH" or self.__fun == "TH":
            y = self.th(y)
        elif self.__fun == "RELU":
            y = self.ReLU(y)
        else:
            raise Exception()

        return y

class layer(object):

    """
    neruals:
    ===
        <list>: <nerual>

    connect_type:
    ===
        <str>: "FULLY_CONNECT" or "SPARSE_CONNECT"
    """

    def __init__( self, neruals = None, connect_type = None, sparse_connection_radius = None ):
        self.__lay = neruals
        self.__connection = connect_type
        self.__radius = sparse_connection_radius


