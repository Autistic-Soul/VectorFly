#!/usr/bin/env Python
# -*- coding: utf-8 -*-

################################################################################################################################################################################################################################################
"""
    Created On 2018.‎11‎.‎2, ‏‎12:36:48
    @author: sandyzikun
"""
################################################################################################################################################################################################################################################

# -*- 导入类库 -*-
try:
    import math
    import cmath
    import numpy as np
    import scipy as sp
    import pandas as pd
except:
    pass

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

# -*- 辅助函数 -*-

def range_except(range_list, except_site):
    _ = list(range_list)
    _.pop(except_site)
    return _

def Hadamard_Multiply(this_matrix, that_matrix):
    if not same_size_matrix(this_matrix, that_matrix):
        raise Exception()
    return [ [ ( this_matrix[i][j] * that_matrix[i][j] ) for j in range(len(this_matrix[0])) ] for i in range(len(this_matrix)) ]

def get_part_matrix(origin_mat, start_point, get_size):
    array = []
    for i in range(start_point[0], start_point[0] + get_size[0]):
        array.append([])
        for j in range(start_point[1], start_point[1] + get_size[1]):
            array[-1].append(origin_mat[i][j])
    return array

def sum_2d(mat):
    if not is_matrix(mat):
        raise Exception()
    return sum([ sum([ mat[i][j] for j in range(len(mat[0])) ]) for i in range(len(mat)) ])

def max_2d(mat):
    if not is_matrix(mat):
        raise Exception()
    return max([ max([ mat[i][j] for j in range(len(mat[0])) ]) for i in range(len(mat)) ])

def max_abs_2d(mat):
    if not is_matrix(mat):
        raise Exception()
    return max([ max([ abs(mat[i][j]) for j in range(len(mat[0])) ]) for i in range(len(mat)) ])

def mean_2d(mat):
    if not is_matrix(mat):
        raise Exception()
    return ( sum([ sum([ mat[i][j] for j in range(len(mat[0])) ]) for i in range(len(mat)) ]) / ( len(mat) * len(mat[0]) ) )

def sum_abs_2d(mat):
    if not is_matrix(mat):
        raise Exception()
    return sum([ sum([ abs(mat[i][j]) for j in range(len(mat[0])) ]) for i in range(len(mat)) ])

def Frobenius_Norm(mat):
    if not is_matrix(mat):
        raise Exception()
    return math.sqrt(sum([ sum([ ( mat[i][j] * mat[i][j] ) for j in range(len(mat[0])) ]) for i in range(len(mat)) ]))

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

    def len(self):
        return len(self.__ARRAY), len(self.__ARRAY[0])

    # -*- 同型矩阵判定 -*-
    def same_matrix(self, other):
        return True if ( len(self.__ARRAY) == len(other.Elements()) and len(self.__ARRAY[0]) == len(other.Elements()[0]) ) else False

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

# -*- 卷积算子定义 -*-
def convolution(convolved_matrix, kernel_matrix):
    convolved_mat = convolved_matrix.Elements() if type(convolved_matrix) == matrix else convolved_matrix
    kernel_mat = kernel_matrix.Elements() if type(kernel_matrix) == matrix else kernel_matrix
    c_size = len(convolved_mat), len(convolved_mat[0])
    k_size = len(kernel_mat), len(kernel_mat[0])
    if k_size[0] > c_size[0] or k_size[1] > c_size[1]:
        raise Exception()
    elif k_size == c_size:
        return Hadamard_Multiply(convolved_mat, kernel_mat)
    array = []
    for i in range(c_size[0] - k_size[0] + 1):
        array.append([])
        for j in range(c_size[1] - k_size[1] + 1):
            part_mat = get_part_matrix(convolved_mat, (i, j), k_size)
            multiplyed_mat = Hadamard_Multiply(part_mat, kernel_mat)
            array[-1].append(sum_2d(multiplyed_mat))
    return array

# -*- 池化算子定义 -*-
def pooling(pooling_matrix, stride, pooling_type = "MAX"):
    pooling_mat = pooling_matrix.Elements() if type(pooling_matrix) == matrix else pooling_matrix
    mat_size = len(pooling_mat), len(pooling_mat[0])
    if mat_size[0] % stride[0] != 0 or mat_size[1] % stride[1] != 0:
        raise Exception()
    array = []
    for i in range(0, mat_size[0] - stride[0] + 1, stride[0]):
        array.append([])
        for j in range(0, mat_size[1] - stride[1] + 1, stride[1]):
            part_mat = get_part_matrix(pooling_mat, (i, j), stride)
            if pooling_type == "MAX":
                array[-1].append(max_2d(part_mat))
            elif pooling_type == "MAX_ABS":
                array[-1].append(max_abs_2d(part_mat))
            elif pooling_type == "MEAN":
                array[-1].append(mean_2d(part_mat))
            elif pooling_type in [ "SUM_ABS", "L1_NORM" ]:
                array[-1].append(sum_abs_2d(part_mat))
            elif pooling_type in [ "FROBENIUS", "L2_NORM" ]:
                array[-1].append(Frobenius_Norm(part_mat))
    return array

# -*- 测试区域(Testing Part)-*-
testing_pooled_mat1 = [ [ 1, 0, 1, 0 ], [ 0, 0, 0, 0 ], [ 1, 0, 1, 0 ], [ 0, 0, 0, 0 ] ]
testing_pooled_mat2 = [ [ 1, 2, 3, 4 ], [ 5, 6, 7, 8 ], [ 9, 10, 11, 12 ], [ 13, 14, 15, 16 ] ]
testing_kernel_mat3 = [ [ 1, 2 ], [ 3, 4 ] ]
testing_convolved_mat4 = [ [ 1, 0, 1, 0 ], [ 0, 0, 0, 0 ], [ 1, 0, 1, 0 ], [ 0, 0, 0, 0 ] ]
print(pooling(testing_pooled_mat1, (2, 2), "MAX"))
print(pooling(testing_pooled_mat1, (2, 2), "MAX_ABS"))
print(pooling(testing_pooled_mat1, (2, 2), "MEAN"))
print(pooling(testing_pooled_mat1, (2, 2), "SUM_ABS"))
print(pooling(testing_pooled_mat1, (2, 2), "FROBENIUS"))
print(pooling(testing_pooled_mat2, (2, 2), "MAX"))
print(pooling(testing_pooled_mat2, (2, 2), "MAX_ABS"))
print(pooling(testing_pooled_mat2, (2, 2), "MEAN"))
print(pooling(testing_pooled_mat2, (2, 2), "SUM_ABS"))
print(pooling(testing_pooled_mat2, (2, 2), "FROBENIUS"))
print(convolution(testing_convolved_mat4, testing_kernel_mat3))
################################################################################################################################################################################################################################################
