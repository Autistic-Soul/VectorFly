#!/usr/bin/env Python
# -*- coding: utf-8 -*-
# - vector -
from random import randint
class _Vector:
    def __init__(self, _Array = [0], _Type = "\'Ln\' or \'Col\'"):
        self.__data = []
        self.__type = ""
        if len(_Array) > 0:
            flag = True
            for each in _Array:
                if (type(each) in [ int, float, complex ]) == False:
                    flag = False
                    break
            if flag == True:
                self.__data = _Array
            else:
                raise RuntimeError()
            del each, flag
            pass
        else:
            raise RuntimeError()
        if _Type == "Ln":
            self.__type = "Ln"
        elif _Type == "Col":
            self.__type = "Col"
        else:
            r = randint(0, 1)
            if r == 0:
                self.__type = "Ln"
            elif r == 1:
                self.__type = "Col"
            del r
        return
    def _Data(self):
        return self.__data
    def _Type(self):
        return self.__type
    def _Trans(self):
        if self.__type == "Col":
            return _Vector(self.__data, "Ln")
        elif self.__type == "Ln":
            return _Vector(self.__data, "Col")
    def __len__(self):
        return len(self.__data)
    def __str__(self):
        _Str = ""
        _Str += "["
        for i in range(len(self.__data)):
            _Str += " "
            _Str += str(self.__data[i])
            if i + 1 != len(self.__data):
                _Str += ","
        _Str += " "
        _Str += "]"
        del i
        return _Str
    def __repr__(self):
        _Str = self.__str__()
        _Str += "\nType: " + self.__type
        _Str += self.__type
        del i
        return _Str
    def __add__(self, other):
        if len(self.__data) != len(other.__data):
            raise RuntimeError()
        elif self.__type != other.__type:
            raise RuntimeError()
        else:
            _Array = []
            for i in range(len(self.__data)):
                _Array.append(self.__data[i] + other.__data[i])
            _Type = self.__type
            return _Vector(_Array = _Array, _Type = _Type)
    def __sub__(self, other):
        if len(self.__data) != len(other.__data):
            raise RuntimeError()
        elif self.__type != other.__type:
            raise RuntimeError()
        else:
            _Array = []
            for i in range(len(self.__data)):
                _Array.append(self.__data[i] - other.__data[i])
            _Type = self.__type
            return _Vector(_Array = _Array, _Type = _Type)
    def __mul__(self, other):
        if type(self) == _Vector and type(other) == _Vector:
            if len(self.__data) != len(other.__data):
                raise RuntimeError()
            elif self.__type != other.__type:
                raise RuntimeError()
            else:
                tot = 0
                for i in range(len(self.__data)):
                    tot += self.__data[i] * other.__data[i]
                del i
                return tot
        elif type(self) == _Vector and type(other) in [ int, float, complex ]:
            _Array = []
            _Type = self.__type
            for each in self.__data:
                _Array.append(other * each)
            del each
            return _Vector(_Array = _Array, _Type = _Type)
        else:
            raise RuntimeError()
alpha = _Vector([ 1, 2, 3, 4, 5])
beta = _Vector([ -1, -2, -3, -4, -5])
