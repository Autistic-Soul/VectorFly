#!/usr/bin/env Python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

from sklearn.svm import SVC

# 线性核函数
def linear_kernel(a, b):
    return np.dot(np.array(a), np.array(b))

# 矢量/向量距离
def vector_distance(a, b):
    return np.linalg.norm(a - b)

# 径向基核函数(高斯核)
def rbf_kernel(a, b, _sigma = 1.):
    return np.exp((-.5) * vector_distance(a, b) / _sigma / _sigma )

# 核函数端口
def kernel_function(a, b, kernel_type = "LINEAR", rbf_sigma = 1.):
    return rbf_kernel(a, b, rbf_sigma) if kernel_type == "RBF" else linear_kernel(a, b)

class SVMClassifier(object):

    # initializations
    def __init__(self, C = 1., kernel_type = "LINEAR", rbf_sigma = 1.):

        self.C = C                      # 参数 C
        self.kernel_type = kernel_type  # 核函数类型
        self.rbf_sigma = rbf_sigma      # 径向基参数 σ

        self.X = None                   # 训练数据 X (输入矢量群/输入向量群)
        self.y = None                   # 训练数据 y (输出纯量群/输出标量群/输出数量群)(标签集)
        self.M = 0                      # 训练集点数 M
        self.m = 0                      # 输入维数 m

        self.alphas = (None, None)      # 拉格朗日乘子矢量/拉格朗日乘子向量α
        self.xis = None                 # 软化间隔差矢量/软化间隔差向量ξ
        self.kernel_mat = None          # 核矩阵 Ker(i, j), 用于存储任意两点间的核函数值
        self.distance_mat = None        # 距离矩阵(类似于OI(信息竞赛)中的邻接矩阵) Dis(i, j), 用于存储任意两点间的闵可夫斯基距离/闵氏距离(在这里为欧几里得距离/欧氏距离)
        self.XPairs_distanced = []      # 按照距离从大到小排列的输入点对(couple)

        self.sv_sites = []              # 输入矢量/输入向量中成为支持矢量/支持向量的下标(序号)列表
        self.sv_X = None                # 支持矢量/支持向量平均值 x_s
        self.sv_y = 0.                  # 支持矢量/支持向量对应标签的平均值 y_s
        self.bias = 0.                  # 偏置 bias(b)

    # re-initializations
    def re_init(self):

        self.X = None                   # 训练数据 X (输入矢量群/输入向量群)
        self.y = None                   # 训练数据 y (输出纯量群/输出标量群/输出数量群)(标签集)
        self.M = 0                      # 训练集点数 M
        self.m = 0                      # 输入维数 m

        self.alphas = (None, None)      # 拉格朗日乘子矢量/拉格朗日乘子向量α
        self.xis = None                 # 软化间隔差矢量/软化间隔差向量ξ
        self.kernel_mat = None          # 核矩阵 Ker(i, j), 用于存储任意两点间的核函数值
        self.distance_mat = None        # 距离矩阵(类似于OI(信息竞赛)中的邻接矩阵) Dis(i, j), 用于存储任意两点间的闵可夫斯基距离/闵氏距离(在这里为欧几里得距离/欧氏距离)
        self.XPairs_distanced = []      # 按照距离从大到小排列的输入点对(couple)

        self.sv_sites = []              # 输入矢量/输入向量中成为支持矢量/支持向量的下标(序号)列表
        self.sv_X = None                # 支持矢量/支持向量平均值 x_s
        self.sv_y = 0.                  # 支持矢量/支持向量对应标签的平均值 y_s
        self.bias = 0.                  # 偏置 bias(b)

    # training
    def fit(self, X, y):

        self.re_init()                                      # 刷新模型

        self.X, self.y = X, y                               # 存储输入&输出数据集
        self.M, self.m = len(X), len(X[0])                  # 存储数据集条数&输入维数

        self.kernel_mat = np.array([ [ kernel_function(X[i], X[j], self.kernel_type, self.rbf_sigma) for j in range(self.M) ] for i in range(self.M) ]) # 构造 Ker(i, j)
        self.distance_mat = np.array([ [ vector_distance(X[i], X[j]) for j in range(self.M) ] for i in range(self.M) ])                                 # 构造 Dis(i, j)
        self.alphas = np.zeros(self.M)                                                                                                                  # 初始化α
        self.xis = np.zeros(self.M)                                                                                                                     # 初始化ξ
        self.sv_X = np.zeros(self.m)                                                                                                                    # 初始化 x_s
        self.sv_y = 0.                                                                                                                                  # 初始化 y_s

        # 根据距离计算输入点对排列
        for i in range(self.M):
            for j in range(self.M):
                self.XPairs_distanced.append((self.distance_mat[i][j], (i, j)))
        self.XPairs_distanced.sort(reverse = True)

        # 汇总所有支持矢量/支持向量
        for _ in range(self.M):
            if self.alphas[_] != 0.:
                self.sv_sites.append(_)

        self.sv_X = sum([ self.X[_] for _ in self.sv_sites ]) / len(self.sv_sites)                                                                                      # 计算 x_s
        self.sv_y = sum([ self.y[_] for _ in self.sv_sites ]) / len(self.sv_sites)                                                                                      # 计算 y_s
        self.bias = self.sv_y - sum([ (self.alphas[_] * self.y[_] * kernel_function(self.sv_X, self.X[_], self.kernel_type, self.rbf_sigma)) for _ in self.sv_sites ])  # 计算 bias

    # predicts
    def __each_predict(self, x):
        return np.sign(sum((self.y[_] * self.alphas[_]) for _ in self.sv_sites) * kernel_function(x, self.sv_X) + self.bias)
    def predict(self, X):
        return [ self.__each_predict(X[_]) for _ in range(len(X)) ]
