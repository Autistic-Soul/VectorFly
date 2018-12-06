#!/usr/bin/env Python
# -*- coding: utf-8 -*-
# - matrix -
def isMat(_Array):
    if type(_Array) != list:
        return False
    elif len(_Array) == 0:
        return False
    else:
        for _Each in _Array:
            if type(_Each) != list:
                del _Each
                return False
            elif len(_Each) == 0:
                del _Each
                return False
            else:
                for _Each2 in _Each:
                    if (type(_Each2) in [ int, float, complex ]) == False:
                        del _Each, _Each2
                        return False
        flag = len(_Array[0])
        for _Each in _Array:
            if len(_Each) != flag:
                del _Each, _Each2
                return False
        del _Each, _Each2
        return True
def isSame(_This, _That):
    if type(_This) != _Matrix or type(_That) != _Matrix:
        raise RuntimeError()
    else:
        if (_This._Len() == _That._Len()):
            return True
        else:
            return False
class _Matrix:
    def __init__(self, _Array = [[0]]):
        if isMat(_Array) == False:
            raise RuntimeError()
        else:
            self.__data = _Array
        pass
    def __len__(self):
        return len(self.__data) * len(self.__data[0])
    def __add__(self, other):
        if isSame(self, other) == False:
            raise RuntimeError()
        else:
            _M, _N = self._Len()
            _Array = []
            for i in range(_M):
                _ARray = []
                for j in range(_N):
                    _ARray.append(self.__data[i][j] + other.__data[i][j])
                _Array.append(_ARray)
            del _ARray, _M, _N
            return _Matrix(_Array = _Array)
    def __sub__(self, other):
        if isSame(self, other) == False:
            raise RuntimeError()
        else:
            _M, _N = self._Len()
            _Array = []
            for i in range(_M):
                _ARray = []
                for j in range(_N):
                    _ARray.append(self.__data[i][j] - other.__data[i][j])
                _Array.append(_ARray)
            del _ARray, _M, _N
            return _Matrix(_Array = _Array)
    def __mul__(self, other):
        if type(self) == _Matrix and type(other) == _Matrix:
            if self._Len("Col") != other._Len("Ln"):
                raise RuntimeError()
            else:
                _M1 = self.__data
                _M2 = other._Trans().__data
                _Array = []
                for i in range(len(_M1)):
                    _ARray = []
                    for j in range(len(_M2)):
                        _Sum = 0
                        for k in range(len(_M1[i])):
                            _Sum += _M1[i][k] * _M2[j][k]
                        _ARray.append(_Sum)
                    _Array.append(_ARray)
                del _ARray, _M1, _M2, i, j, k
                return _Matrix(_Array = _Array)
        elif type(self) == _Matrix and type(other) in [ int, float, complex ]:
            _Array = []
            for _Each in self.__data:
                _ARray = []
                for _Each2 in _Each:
                    _ARray.append(other * _Each2)
                _Array.append(_ARray)
            del _ARray, _Each, _Each2
            return _Matrix(_Array = _Array)
    def __pow__(self, _K):
        if self._Len("Ln") != self._Len("Col"):
            raise RuntimeError()
        else:
            if _K > 0:
                return ( self ** (_K - 1) ) * self
            else:
                _Array = []
                for i in range(self._Len("Ln")):
                    _ARray = []
                    for j in range(self._Len("Col")):
                        if i == j:
                            _ARray.append(1)
                        else:
                            _ARray.append(0)
                    _Array.append(_ARray)
                del _ARray, i, j
                return _Matrix(_Array = _Array)
    def __str__(self):
        _Str = "[\n"
        for i in range(len(self.__data)):
            _Str += "\t["
            for j in range(len(self.__data[i])):
                _Str += " " + str(self.__data[i][j])
                if j + 1 != len(self.__data[i]):
                    _Str += ","
                else:
                    _Str += " ]"
                    if i + 1 != len(self.__data):
                        _Str += ",\n"
                    else:
                        _Str += "\n"
        _Str += "]\n"
        del i, j
        return _Str
    def __repr__(self):
        _Str = self.__str__()
        _Str += "Shape: ( " + str(self._Len("Ln")) + ", " + str(self._Len("Col")) + " )"
        return _Str
    def _Data(self):
        return self.__data
    def _Trans(self):
        _Array = []
        for i in range(len(self.__data[0])):
            _Array.append([])
            pass
        for i in range(len(self.__data)):
            for j in range(len(self.__data[i])):
                _Array[j].append(self.__data[i][j])
        del i, j
        return _Matrix(_Array = _Array)
    def _Len(self, _Type = "\'Ln\' or \'Col\'"):
        if _Type == "Ln":
            return len(self.__data)
        elif _Type == "Col":
            return len(self.__data[0])
        else:
            return ( len(self.__data), len(self.__data[0]) )
def _Diag_Matrix(_ARray = [0]):
    _Array = []
    for i in range(len(_ARray)):
        _ARRay = []
        for j in range(len(_ARray)):
            if i == j:
                _ARRay.append(_ARray[i])
            else:
                _ARRay.append(0)
        _Array.append(_ARRay)
    del i, j, _ARRay
    return _Matrix(_Array = _Array)
def _E_Matrix(_Elements = 1, _Step = 1):
    _ARray = []
    for i in range(_Step):
        _ARray.append(_Elements)
    del i
    return _Diag_Matrix(_ARray = _ARray)
_Mats = []
# matrix0
_Mats.append(
    _Matrix(
        [
            [ 246, 427, 327 ],
            [ 1014, 543, 443 ],
            [ (-342), 721, 621 ]
            ]
        )
    )
# matrix1
_Mats.append(
    _Matrix(
        [
            [ 0.67, 0.66 ],
            [ 0.33, 0.34 ],
            ]
        )
    )
# matrix2
_Mats.append(
    _Matrix(
        [
            [ 800 ],
            [ 200 ]
            ]
        )
    )
# matrix3
_Mats.append(
    _Matrix(
        [
            [ 5, 1, 0 ],
            [ 0, 5, 2 ],
            [ 0, 0, 5 ]
            ]
        )
    )
# matrix4
_Mats.append(
    _Matrix(
        [
            [ 2, 4, (-6) ],
            [ 1, 2, (-3) ],
            [ 4, 8, (-12) ]
            ]
        )
    )
# matrix5
_Mats.append(
    _Matrix(
        [
            [ 2 ],
            [ 1 ],
            [ 4 ]
            ]
        )
    )
# matrix6
_Mats.append(
    _Matrix(
        [
            [ 1, 2, (-3) ]
            ]
        )
    )
# matrix7
_Mats.append(
    _Matrix(
        [
            [ 0, 1, 0 ],
            [ 1, 0, (-1) ],
            [ 0, (-1), 0 ]
            ]
        )
    )
# matrix8
_Mats.append(
    _Matrix(
        [
            [ 9, 0, (-6) ],
            [ 0, 15, 0 ],
            [ 0, 0, 21 ]
            ]
        )
    )
# matrix9
_Mats.append(
    _Matrix(
        [
            [ 1, 0, (-1) ],
            [ 0, 2, 0 ],
            [ 0, 0, 3 ]
            ]
        )
    )
# matrix10
_Mats.append(
    _Matrix(
        [
            [ 2, 1, 1 ],
            [ 1, 2, 1 ],
            [ 1, 1, 2 ]
            ]
        )
    )
# matrix11
_Mats.append(
    _Matrix(
        [
            [ 0.75, (-0.25), (-0.25) ],
            [ (-0.25), 0.75, (-0.25) ],
            [ (-0.25), (-0.25), 0.75 ]
            ]
        )
    )
