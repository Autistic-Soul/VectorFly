#!/usr/bin/env Python
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

M_MAX_ITERATION_TIMES = 1000
DEFAULT_MAX_STEP = 0.
DEFAULT_SECOND_ALPHA = 0.
DEFAULT_SECOND_ERROR = 0.

def vector_distance2(a, b):
    return np.linalg.norm(a - b) ** 2

def rbf_kernel_function(a, b, rbf_sigma = 1.):
    return np.exp(  vector_distance2(a, b) * (-.5) / rbf_sigma**2   )

def kernel_function(a, b, kernel_options):
    return rbf_kernel_function(a, b, kernel_options[-1]) if kernel_options[0]=="RBF" else np.dot(a, b)

# -*- Support Vector Machine -*-
# -*- 支持矢量机 -*-
# -*- 支持向量机 -*-
class SVMClassifier:

    def __init__(self, C = 1., tolerance = .001, kernel_options = ("RBF", 1.)):
        self.C = C
        self.tol = tolerance
        self.ker_opt = kernel_options
        return

    # Initializing before Training
    # 训练预处理
    def __fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.M = len(X)
        self.m = len(X[0])
        self.alphas = np.zeros(self.M)                                                                                                  # Initializing Alpha(s)
        self.Kernel_Matrix = np.mat([ [ kernel_function(X[i], X[j], self.ker_opt) for j in range(self.M) ] for i in range(self.M) ])    # Computing Kernel-Matrix
        self.Error_Array = np.mat(np.zeros([ self.M, 2 ]))                                                                              # Preparing Error-Matrix
        self.Bias = 0.                                                                                                                  # Preparing Bias
        self.trainable = True
        return

    # Computing Error
    # 计算误差
    def __compute_error(self, alpha1st):
        pred = np.mat(self.alphas * self.y) * self.Kernel_Matrix[ : , alpha1st ] + self.Bias
        return np.float64( pred - self.y[alpha1st] )

    # Updating Error
    # 更新误差
    def __update_error(self, alpha1st):
        self.Error_Array[alpha1st][0] = 1
        self.Error_Array[alpha1st][1] = self.__compute_error(alpha1st)
        return

    # Undating All Error
    # 更新全部误差
    def __update_all_error(self):
        for _ in range(self.M):
            self.__update_error(_)
        return

    # Selecting the Second Alpha
    # 挑选第二个拉格朗日乘子分量
    def __inner_loop(self, alpha1st, error1st):

        # 标记为该拉格朗日乘子分量已经被优化过
        self.Error_Array[alpha1st][0] = 1
        self.Error_Array[alpha1st][1] = error1st
        candidate_alphas = np.nonzero(self.Error_Array[ : , 0 ].A)[0]

        max_step = DEFAULT_MAX_STEP
        alpha2nd = DEFAULT_SECOND_ALPHA
        error2nd = DEFAULT_SECOND_ERROR

        if len(candidate_alphas) > 1:
            for alpha3rd in candidate_alphas:
                if alpha3rd == alpha1st:
                    continue
                error3rd = self.__compute_error(alpha3rd)
                if np.abs(error3rd - error1st) > max_step:
                    max_step = np.abs(error3rd - error1st)
                    alpha2nd, error2nd = alpha3rd, error3rd

        # 随机选择
        else:
            alpha2nd = alpha1st
            while alpha2nd == alpha1st:
                alpha2nd = np.random.randint(self.M)
            error2nd = self.__compute_error(alpha2nd)

        return alpha2nd, error2nd

    # SMO核心部分
    def __choose_update(self, alpha1st):

        error1st = self.__compute_error(alpha1st)

        # 判断选择出的第一个变量是否违背了KKT条件
        if ( self.y[alpha1st] * error1st < -self.tol and self.alphas[alpha1st] < self.C ) or ( self.y[alpha1st] * error1st > self.tol and self.alphas[alpha1st] > 0 ):

            # 1. 选择第二个分量
            alpha2nd, error2nd = self.__inner_loop(alpha1st, error1st)
            alpha1st_old = self.alphas[alpha1st].copy()
            alpha2nd_old = self.alphas[alpha2nd].copy()

            # 2. 计算上下界
            L = np.max([ 0, self.alphas[alpha2nd] - self.alphas[alpha1st] if self.y[alpha1st] != self.y[alpha2nd] else self.alphas[alpha1st] + self.alphas[alpha2nd] - self.C ])        # 下界
            H = np.min([ self.C, self.alphas[alpha2nd] - self.alphas[alpha1st] + self.C if self.y[alpha1st] != self.y[alpha2nd] else self.alphas[alpha2nd] - self.alphas[alpha1st] ])   # 上界
            if L == H:
                return 0

            # 3. 计算学习率(Learning Rate)
            learning_rate = (-1.) * ( self.Kernel_Matrix[alpha1st, alpha1st] - self.Kernel_Matrix[alpha1st, alpha2nd] - self.Kernel_Matrix[alpha2nd, alpha1st] + self.Kernel_Matrix[alpha2nd, alpha2nd] )

            # 4. 更新 alpha2nd
            self.alphas[alpha2nd] -= self.y[alpha2nd] * (error1st - error2nd) / learning_rate

            # 5. 确定最终的 alpha2nd
            self.alphas[alpha2nd] = np.min([ self.alphas[alpha2nd], H ])
            self.alphas[alpha2nd] = np.max([ self.alphas[alpha2nd], L ])

            # 6. 判断是否结束
            if np.abs(alpha2nd_old - self.alphas[alpha2nd]) < self.tol:
                self.__update_error(alpha2nd)
                return 0

            # 7. 更新 alpha1st
            self.alphas[alpha1st] += self.y[alpha1st] * self.y[alpha2nd] * (alpha2nd_old-self.alphas[alpha2nd])

            # 8. 更新 bias
            bias1 = self.Bias - error1st \
                  - self.y[alpha1st] * (self.alphas[alpha1st]-alpha1st_old) * self.Kernel_Matrix[alpha1st, alpha1st] \
                  - self.y[alpha2nd] * (self.alphas[alpha2nd]-alpha2nd_old) * self.Kernel_Matrix[alpha1st, alpha2nd]
            bias2 = self.Bias - error2nd \
                  - self.y[alpha1st] * (self.alphas[alpha1st]-alpha1st_old) * self.Kernel_Matrix[alpha2nd, alpha1st] \
                  - self.y[alpha2nd] * (self.alphas[alpha2nd]-alpha2nd_old) * self.Kernel_Matrix[alpha2nd, alpha2nd]
            if 0 < self.alphas[alpha1st] < self.C:
                self.Bias = bias1
            elif 0 < self.alphas[alpha2nd] <self.C:
                self.Bias = bias2
            else:
                self.Bias = .5 * (bias1 + bias2)

            # 9. 更新 error
            self.__update_error(alpha1st)
            self.__update_error(alpha2nd)

            return 1

        # 如果选择出的第一个变量没有违背了KKT条件, 直接退出函数
        else:
            return 0

    def train(self, X, y, m_max_iter = M_MAX_ITERATION_TIMES, print_details = True):

        # -*- Initializing(初始化) -*-
        self.__fit(X, y)

        # -*- Training(训练开始) -*-
        if self.trainable:

            to_entire_set = True
            alpha_couples_changed = 0
            iteration_times = 0

            # -*- Iterating(迭代开始) -*-
            while iteration_times < m_max_iter and ( alpha_couples_changed > 0 or to_entire_set ):

                if print_details:
                    print("\tITER[%d]:" % iteration_times)

                if to_entire_set:
                    # For All X_Train(s)
                    # 对所有的输入数据
                    for _ in range(self.M):
                        alpha_couples_changed += self.__choose_update(_)

                else:
                    # For Support-Vectors
                    # 对支持矢量/支持向量
                    sv_sites = []
                    for _ in range(self.M):
                        if self.alphas[_] > 0 and self.alphas[_] < C:
                            sv_sites.append(_)

                    for _ in sv_sites:
                        alpha_couples_changed += self.__choose_update(_)

                iteration_times += 1

                # 在所有的输入数据与支持矢量/支持向量之间交替
                to_entire_set = True if alpha_couples_changed==0 else False

            self.trainable = False

        return


## TESTING PART
