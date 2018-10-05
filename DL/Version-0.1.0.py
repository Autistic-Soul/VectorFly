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
        <str>: "SIGMOID" or "TANH" ( or "TH" ) or "RELU"
    """

    def __init__( self, weights = None, bias = None, activation_function = "TANH" ):
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

    def Compute( self, X = None ):

        if len(self.__W) != len(X):
            raise Exception()

        if self.__fun == "SIGMOID":
            return self.Sigmoid(sum([ self.__W[i] * X[i] for i in range(len(self.__W)) ]) + self.__B)

        elif self.__fun in [ "TANH", "TH" ]
            return self.th(sum([ self.__W[i] * X[i] for i in range(len(self.__W)) ]) + self.__B)

        elif self.__fun == "RELU":
            return self.ReLU(sum([ self.__W[i] * X[i] for i in range(len(self.__W)) ]) + self.__B)

        else:
            raise Exception()

class layer(object):

    """
    neruals:
    ===
        <list>: <nerual>

    connect_type:
    ===
        <str>: "FULLY_CONNECT" or "SPARSE_CONNECT"
    """

    def __init__( self, neruals = None, connect_type = "FULLY_CONNECT", sparse_connection_radius = None ):
        self.__nodes = neruals
        self.__connection = connect_type
        self.__radius = sparse_connection_radius

    def Compute( self, X ):

        if self.__connection == "FULLY_CONNECT":
            return [ each_nerual.Compute(X) for each_nerual in self.__nodes ]

        elif self.__connection == "SPARSE_CONNECT":
            raise Exception()

        else:
            raise Exception()

class net(object):

    def __init__( self, layers = None ):
        self.__lays = layers
