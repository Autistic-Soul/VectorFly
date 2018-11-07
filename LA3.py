#!/usr/bin/env Python
# -*- coding: utf-8 -*-

################################################################################################################################################################################################################################################
"""
    Created On 2018.‎11‎.‎2, ‏‎12:36:48
    @author: sandyzikun
"""
################################################################################################################################################################################################################################################

# -*- 导入类库 -*-
import math
import cmath
try:
    import numpy as np
    from numpy import linalg as npla
except ImportError as _:
    print(_)
except Exception as _:
    print(_)
try:
    import scipy as sp
except ImportError as _:
    print(_)
except Exception as _:
    print(_)
try:
    import pandas as pd
except ImportError as _:
    print(_)
except Exception as _:
    print(_)

# -*- 辅助函数 -*-

def range_except(range_list, except_site):
    _ = list(range_list)
    _.pop(except_site)
    return _

# -*- 判定函数 -*-

def is_matrix(mat):
    if type(mat) != list or len(mat) == 0 or type(mat[0]) != list or len(mat[0]) == 0:
        return False
    for _ in range(1, len(mat)):
        if type(mat[_]) != list or len(mat[_]) != len(mat[_ - 1]):
            return False
    return True

def is_square_matrix(mat):
    return True if is_matrix(mat) and len(mat) == len(mat[0]) else False

def is_L_matrix(mat):
    if not is_square_matrix(mat):
        return False
    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            if mat[i][j] != 0:
                return False
    return True

def is_U_matrix(mat):
    if not is_square_matrix(mat):
        return False
    for i in range(len(mat)):
        for j in range(i):
            if mat[i][j] != 0:
                return False
    return True

def same_size_matrix(this_matrix, that_matrix):
    if not ( is_matrix(this_matrix) and is_matrix(that_matrix) ):
        return False
    return True if len(this_matrix) == len(that_matrix) and len(this_matrix[0]) == len(that_matrix[0]) else False

# -*- 奥义 * 四重递归 -*-

def cofactor_matrix(mat, MATxy):
    if not is_matrix(mat):
        raise Exception()
    _ = []
    for i in range_except(range(len(mat)), MATxy[0]):
        _.append([])
        for j in range_except(range(len(mat[0])), MATxy[1]):
            _[-1].append(mat[i][j])
    return _

def cofactor_value(mat, MATxy):
    if not is_square_matrix(mat):
        raise Exception()
    return determinant(cofactor_matrix(mat, MATxy))

def alg_cofactor(mat, MATxy):
    if not is_square_matrix(mat):
        raise Exception()
    return cofactor_value(mat, MATxy) * ( (-1) ** sum(MATxy) )

def determinant(mat):
    if not is_square_matrix(mat):
        raise Exception()
    elif len(mat) == 1:
        return mat[0][0]
    return sum([ ( mat[_][0] * alg_cofactor(mat, (_, 0)) ) for _ in range(len(mat)) ])

# -*- 矩阵定义 -*-

class matrix(object):

    def __init__(self, array = [[]]):
        if not is_matrix(array):
            raise Exception("The Array You\'ve Just Input Is Not A Matrix!")
        self.__ARRAY = array

    def __repr__(self):
        _ =                                     "Matrix:[\n"
        Max_Strlen = max([ max([ len(str(self.__ARRAY[i][j])) for j in range(len(self.__ARRAY[0])) ]) for i in range(len(self.__ARRAY)) ])
        for i in range(len(self.__ARRAY)):
            _ +=                                "  ["
            for j in range(len(self.__ARRAY[0])):
                _ +=                            " " * (Max_Strlen - len(str(self.__ARRAY[i][j])))
                _ +=                            " %s" % str(self.__ARRAY[i][j])
                _ +=                            " " if j == len(self.__ARRAY[0]) - 1 else ","
            _ +=                                "]\n"
        _ +=                                    "]\n"
        return _

    def Elements(self):
        return self.__ARRAY

    # -*- 同型矩阵判定 -*-
    def same_matrix(self, other):
        return True if ( len(self.__ARRAY) == len(other.Elements()) and len(self.__ARRAY[0]) == len(other.Elements()[0]) ) else False

    def len(self):
        return len(self.__ARRAY), len(self.__ARRAY[0])

    # -*- 逐元素对应乘积(Hadamard) -*-
    def Hadamard(self, other):
        return matrix([ [ ( self.__ARRAY[i][j] * other.Elements()[i][j] ) for j in range(len(self.__ARRAY[0])) ] for i in range(len(self.__ARRAY)) ])

    # -*- 定义线性运算 -*-

    def __add__(self, other):
        if not self.same_matrix(other):
            raise Exception()
        return matrix([ [ ( self.__ARRAY[i][j] + other.__ARRAY[i][j] ) for j in range(len(self.__ARRAY[0])) ] for i in range(len(self.__ARRAY)) ])

    def __sub__(self, other):
        if not self.same_matrix(other):
            raise Exception()
        return matrix([ [ ( self.__ARRAY[i][j] - other.__ARRAY[i][j] ) for j in range(len(self.__ARRAY[0])) ] for i in range(len(self.__ARRAY)) ])

    def __mul__(self, other):
        if not type(other) == matrix:
            return matrix([ [ ( self.__ARRAY[i][j] * other ) for j in range(len(self.__ARRAY[0])) ] for i in range(len(self.__ARRAY)) ])

    # -*- 定义矩阵乘法 -*-
        else:
            if len(self.__ARRAY) != len(other.__ARRAY[0]):
                raise Exception("The Matrices You\'ve Just Input Cannot Be Multiplyed!")
            return matrix([ [ sum([ ( self.__ARRAY[i][k] * other.__ARRAY[k][j] ) for k in range(len(self.__ARRAY[0])) ]) for j in range(len(other.__ARRAY[0])) ] for i in range(len(self.__ARRAY)) ])

    # -*- 定义矩阵乘方 -*-
    def __pow__(self, other):
        if type(other) != int:
            raise Exception()
        elif len(self.__ARRAY) != len(self.__ARRAY[0]):
            raise Exception("The Matrix You Just Input Is Not A Square Matrix!")
        elif other < 0:
            raise Exception("Sorry, We Can\'t Compute Inverse Matrix Yet!")
        mat = self
        for _ in range(1, other):
            mat *= self
        return mat

    # -*- 求解行列式 -*-
    def Determinant(self):
        return determinant(self.Elements())
    def DET(self):
        return self.Determinant()

    # -*- 求解秩 -*-
    def Rank(self):
        return #!
    def R(self):
        return self.Rank()

    # -*- 求解迹 -*-
    def Trace(self):
        if len(self.__ARRAY) != len(self.__ARRAY[0]):
            raise Exception("The Matrix You Just Input Is Not A Square Matrix!")
        product = 1
        for _ in range(len(self.__ARRAY)):
            product *= self.__ARRAY[_][_]
        return product
    def TR(self):
        return self.Trace()

    # -*- 定义转置 -*-
    def Transposition(self):
        return matrix([ [ self.__ARRAY[j][i] for j in range(len(self.__ARRAY)) ] for i in range(len(self.__ARRAY[0])) ])
    def T(self):
        return self.Transposition()
    def Self_T(self):
        self.__ARRAY = self.T().Elements()
        return

    # -*- 定义副对角线转置 -*-
    def Transposition2(self):
        return matrix([ [ self.__ARRAY[len(self.__ARRAY) - 1 - j][len(self.__ARRAY[0]) - 1 - i] for j in range(len(self.__ARRAY)) ] for i in range(len(self.__ARRAY[0])) ])
    def T2(self):
        return self.Transposition2()
    def Self_T2(self):
        self.__ARRAY = self.T2().Elements()
        return

    # -*- 定义镜像运算 -*-

    def Mirror_Ln(self):
        return matrix([ self.__ARRAY[i] for i in range(len(self.__ARRAY) - 1, -1, -1) ])

    def Mirror_Col(self):
        return matrix([ [ self.__ARRAY[i][j] for j in range(len(self.__ARRAY[0]) - 1, -1, -1) ] for i in range(len(self.__ARRAY)) ])

    def Mirror_2d(self):
        return matrix([ [ self.__ARRAY[i][j] for j in range(len(self.__ARRAY[0]) - 1, -1, -1) ] for i in range(len(self.__ARRAY) - 1, -1, -1) ])

    # -*- 定义初等变换 -*-
    def Ln_Swap(self, Site_A, Site_B):
        array = self.__ARRAY
        array[Site_A], array[Site_B] = array[Site_B], array[Site_A]
        return matrix(array)

    def Col_Swap(self, Site_A, Site_B):
        array = self.__ARRAY
        for _ in range(len(array)):
            array[_][Site_A], array[_][Site_B] = array[_][Site_B], array[_][Site_A]
        return matrix(array)

    def Ln_Mul(self, Site, k):
        if k == 0:
            raise Exception()
        array = self.__ARRAY
        for _ in range(len(array[0])):
            array[Site][_] *= k
        return matrix(array)

    def Col_Mul(self, Site, k):
        if k == 0:
            raise Exception()
        array = self.__ARRAY
        for _ in range(len(array)):
            array[_][Site] *= k
        return matrix(array)

    def Ln_Add(self, Site_B, Site_A, k): # B += k * A
        array = self.__ARRAY
        for _ in range(len(array[0])):
            array[Site_B][_] += array[Site_A][_] * k
        return matrix(array)

    def Col_Add(self, Site_B, Site_A, k):
        array = self.__ARRAY
        for _ in range(len(array)):
            array[_][Site_B] += array[_][Site_A] * k
        return matrix(array)

    # -*- 返回矢量数组 -*-
    def vector_array(self):
        array = self.Elements()
        if len(array) == 1:
            return array[0]
        elif len(array[0]) == 1:
            return [ array[_][0] for _ in range(len(array)) ]
        else:
            raise Exception()

# -*- 对角矩阵 -*-
def diag_matrix(diag_vector):
    array = []
    for i in range(len(diag_vector)):
        array.append([])
        for j in range(len(diag_vector)):
            array[-1].append(diag_vector[i] if i == j else 0)
    return matrix(array)

# -*- 副对角矩阵 -*-
def diag_matrix2(diag_vector):
    array = []
    for i in range(len(diag_vector)):
        array.append([])
        for j in range(len(diag_vector)):
            array[-1].append(diag_vector[i] if i + j + 1 == len(diag_vector) else 0)
    return matrix(array)

# -*- 标量矩阵/数量矩阵/纯量矩阵 -*-
def scalar_matrix(scalar_value, step):
    return diag_matrix([scalar_value] * step)

# -*- 副对角标量矩阵/数量矩阵/纯量矩阵 -*-
def scalar_matrix2(scalar_value, step):
    return diag_matrix2([scalar_value] * step)

# -*- 零矩阵 -*-
def O(step):
    return matrix([ [ 0 for j in range(step) ] for i in range(step) ])

# -*- 单位矩阵 -*-
def E(step):
    return scalar_matrix(1, step)
def I(step):
    return scalar_matrix(1, step)
def eye(step):
    return scalar_matrix(1, step)
def identity_matrix(step):
    return scalar_matrix(1, step)

# -*- 副对角单位矩阵 -*-
def E2(step):
    return scalar_matrix2(1, step)
def I2(step):
    return scalar_matrix2(1, step)
def eye2(step):
    return scalar_matrix2(1, step)
def identity_matrix2(step):
    return scalar_matrix2(1, step)

# -*- 行矩阵/行矢量/行向量 -*-
def Ln_Vector(vector_value):
    return matrix([vector_value])

# -*- 列矩阵/列矢量/列向量 -*-
def Col_Vector(vector_value):
    return matrix([ [vector_value[_]] for _ in range(len(vector_value)) ])

# -*- 定义初等矩阵 -*-

def Ln_Swap_Matrix(step, Site_A, Site_B):
    return eye(step).Ln_Swap(Site_A, Site_B)

def Col_Swap_Matrix(step, Site_A, Site_B):
    return eye(step).Col_Swap(Site_A, Site_B)

def Ln_Mul_Matrix(step, Site, k):
    return eye(step).Ln_Mul(Site, k)

def Col_Mul_Matrix(step, Site, k):
    return eye(step).Col_Mul(Site, k)

def Ln_Add_Matrix(step, Site_B, Site_A, k):
    return eye(step).Ln_Add(Site_B, Site_A, k)

def Col_Add_Matrix(step, Site_B, Site_A, k):
    return eye(step).Col_Add(Site_B, Site_A, k)

# -*- 下三角方程组求解 -*-
def solve_L(L, b):
    A = L.Elements() if type(L) == matrix else L
    ans = b.vector_array() if type(b) == matrix else b
    if not ( len(A) == len(ans) and is_L_matrix(A) ):
        raise Exception()
    for i in range(len(A)):
        ans[i] /= A[i][i]
        if i == len(A) - 1:
            break
        for j in range(i + 1, len(A)):
            ans[j] -= ans[i] * A[j][i]
    return ans

# -*- 上三角方程组求解 -*-
def solve_U(U, b):
    A = U.Elements() if type(U) == matrix else U
    ans = b.vector_array() if type(b) == matrix else b
    if not ( len(A) == len(ans) and is_U_matrix(A) ):
        raise Exception()
    for i in range(len(A) - 1, -1, -1):
        ans[i] /= A[i][i]
        if i == 0:
            break
        for j in range(i):
            ans[j] -= ans[i] * A[j][i]
    return ans

# -*- 矩阵卷积运算 -*-
def convolution(convolved_matrix, kernel_matrix, stride = 1, padding = None):
    if padding != None:
        raise Exception("Sorry, We Can\'t Pad Elements Yet!")

# -*- TESTING PART -*-
testing_matrix_201811031022 = [
    [ 1, 2, 3 ],
    [ 4, 5, 6 ],
    [ 7, 8, 9 ]
    ]
testing_matrix_201811031136 = [
    [ 1, 2 ],
    [ 3, 4 ]
    ]
testing_matrix_201811051304 = [
    [ 1, 2 ],
    [ 3, 4 ]
    ]
testing_matrix_201811061611 = [
    [ 1, 0, 0, 0 ],
    [ 2, 1, 0, 0 ],
    [ 3, 2, 1, 0 ],
    [ 4, 3, 2, 1 ]
    ]
testing_matrix_201811061613 = [
    [ 1, 2, 3, 4 ],
    [ 0, 1, 2, 3 ],
    [ 0, 0, 1, 2 ],
    [ 0, 0, 0, 1 ]
    ]
mat1 = [
    [ 1, 2 ],
    [ 3, 4 ]
    ]
mat2 = [
    [ 1, 2 ],
    [ 3, 4 ]
    ]
################################################################################################################################################################################################################################################
