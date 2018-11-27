#!/usr/bin/env Python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

from sklearn.svm import SVC

def linear_kernel(a, b):
    return np.dot(np.array(a), np.array(b))

def rbf_kernel(a, b, sigma = 1.0):
    delta_vector = a - b
    delta_vector_l2_norm = np.linalg.norm(delta_vector)
    coef_sigma = (-2) * sigma ** 2
    return np.exp(delta_vector_l2_norm / coef_sigma)

def kernel_function(a, b, kernel_type = "LINEAR", rbf_sigma = 1.0):
    return rbf_kernel(a, b, rbf_sigma) if kernel_type == "RBF" else linear_kernel(a, b)

def vector_distance(a, b):
    return np.linalg.norm(a - b)

class SVMClassifier(object):

    def __init__(self, C = 1.0, kernel_type = "LINEAR", rbf_sigma = 1.0):

        self.C = C
        self.kernel_type = kernel_type
        self.rbf_sigma = rbf_sigma

        self.X = None
        self.y = None
        self.M = 0.0
        self.m = 0.0
        self.alpha_vector = (None, None)
        self.soft_margin_vector = None
        self.kerenl_mat = None
        self.distance_mat = None
        self.avg_support_X_vector = None
        self.avg_support_y = 0.0
        self.support_X_vectors_site = []
        self.bias = 0.0

        return

    def re_init(self):

        self.X = None
        self.y = None
        self.M = 0.0
        self.m = 0.0
        self.alpha_vector = (None, None)
        self.soft_margin_vector = None
        self.kerenl_mat = None
        self.distance_mat = None
        self.avg_support_X_vector = None
        self.avg_support_y = 0.0
        self.support_X_vectors_site = []
        self.bias = 0.0

        return

    def fit(self, X, y, C = "AUTO", kernel_type = "AUTO", rbf_sigma = "AUTO"):

        self.re_init()

        self.X, self.y = X, y
        self.M, self.m = len(X), len(X[0])

        self.alpha_vector = np.zeros(self.M)
        self.soft_margin_vector = np.zeros(self.M)
        self.kerenl_mat = np.zeros([ self.M, self.M ])
        self.distance_mat = np.zeros([ self.M, self.M ])
        self.avg_support_X_vector = np.zeros(self.m)
        self.avg_support_y = 0.0

        for i in range(self.M):
            for j in range(self.M):
                self.kerenl_mat[i][j] = kernel_function(X[i], X[j], self.kernel_type, self.rbf_sigma)

        for i in range(self.M):
            for j in range(self.M):
                self.distance_mat[i][j] = vector_distance(X[i], X[j])

        # Add Computing alpha_vectors!

        for _ in range(self.M):
            if self.alpha_vector[_] != 0:
                self.support_X_vectors_site.append(_)
        self.avg_support_X_vector = sum([ self.X[_] for _ in self.support_X_vectors_site ]) / len(self.support_X_vectors_site)
        self.avg_support_y = sum([ self.y[_] for _ in self.support_X_vectors_site ]) / len(self.support_X_vectors_site)
        self.bias = self.avg_support_y - sum([ (self.alpha_vector[_] * self.y[_] * kernel_function(self.avg_support_X_vector, self.X[_], self.kernel_type, self.rbf_sigma)) for _ in self.support_X_vectors_site ])

    def __each_predict(self, x):
        return np.sign(sum((self.y[_] * self.alpha_vector[_]) for _ in self.support_X_vectors_site) * kernel_function(x, self.avg_support_X_vector) + self.bias)

    def predict(self, X):
        return [ self.__each_predict(X[_]) for _ in range(len(X)) ]
