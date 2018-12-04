#!/usr/bin/env Python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

class LinearSVMClassifier(object):

    def __init__(self):

        self.X = None
        self.y = None
        self.M = 0
        self.m = 0

        self.alphas = None
        self.kermat = None
        self.dismat = None

        self.sv_sites = []
        self.sv_X = None
        self.sv_y = 0.
        self.bias = 0.

    def fit(self, X, y):

        self.X = X
        self.y = y

        self.M = len(self.X)
        self.m = len(self.X[0])

        self.kermat = np.array([ [ np.dot(self.X[i], self.X[j]) for j in range(self.M) ] for i in range(self.M) ])
        self.dismat = np.array([ [ None for j in range(self.M) ] for i in range(self.M) ])

        return

    def __each_predict(self, x):
        return np.sign()

    def predict(self, X):
        return [ self.__each_predict(_) for _ in X ]
