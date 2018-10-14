#!/usr/bin/env Python
# -*- coding: utf-8 -*-


NUMERIC_TYPES = [ int, float, complex ]


def isMat(_Array = [[0]]):
    if type(_Array) != list:
        return ( False, "" )
    elif len(_Array) == 0:
        return ( False, "" )
    _Standard_Length = len(_Array[0])
    if _Standard_Length == 0:
        return ( False, "" )
    for _Each_Line in _Array:
        if type(_Each_Line) != list:
            return ( False, "" )
        elif len(_Each_Line) != _Standard_Length:
            return ( False, "" )
        for _Each_Element in _Each_Line:
            if ( type(_Each_Element) in NUMERIC_TYPES ) == False:
                return ( False, "" )
    return ( True, "" )


def isVec(_Array = [0]):
    if type(_Array) != list:
        return ( False, "" )
    elif len(_Array) == 0:
        return ( False, "" )
    for _Each_Element in _Array:
        if ( type(_Each_Element) in NUMERIC_TYPES ) == False:
            return ( False, "" )
    return ( True, "" )


def cofactor_matrix( _Array = [[0]], Ln_from_zero = None, Col_from_zero = None ):
    _ARray = []
    for i in range(len(_Array)):
        _Vec = []
        for j in range(len(_Array)):
            _Vec.append(_Array[i][j])
            if j == Col_from_zero:
                _Vec.pop()
        _ARray.append(_Vec)
        if i == Ln_from_zero:
            _ARray.pop()
    return _ARray

def cofactor( _Array = [[0]], Ln_from_zero = None, Col_from_zero = None ):
    return list_determinant(cofactor_matrix( _Array = _Array, Ln_from_zero = Ln_from_zero, Col_from_zero = Col_from_zero ))


def algebraric_cofactor( _Array = [[0]], Ln_from_zero = None, Col_from_zero = None ):
    return ( cofactor( _Array = _Array, Ln_from_zero = Ln_from_zero, Col_from_zero = Col_from_zero ) * ( (-1) ** ( Ln_from_zero + Col_from_zero ) ) )


def list_determinant(_Array = [[0]]):

    if isMat(_Array) == False:
        raise Exception()

    elif len(_Array) != len(_Array[0]):
        raise Exception()

    if len(_Array) == 1 and len(_Array[0]) == 1:
        return _Array[0][0]

    else:
        return sum([ _Array[i][0] * algebraric_cofactor( _Array = _Array, Ln_from_zero = i, Col_from_zero = 0 ) for i in range(len(_Array)) ])


class _Matrix(object):

    def __init__(self, _Array = [[0]]):
        if isMat(_Array = _Array)[0] == False:
            raise Exception(isMat(_Array = _Array)[1])
        self.__data = _Array

    def __len__(self):
        return ( len(self.__data) * len(self.__data[0]) )

    def _Length(self):
        return ( len(self.__data), len(self.__data[0]) )
    def _Len(self):
        return self._Length()

    def __str__(self):
        _Returning_String = "[\n"
        for i in range(self._Length()[0]):
            _Returning_String += ( " " * 4 + "[ " )
            for j in range(self._Length()[1]):
                _Returning_String += str(self.__data[i][j])
                if j == self._Length()[1] - 1:
                    _Returning_String += " ]\n"
                else:
                    _Returning_String += ", "
        _Returning_String += "]\n"
        return _Returning_String

    def __repr__(self):
        _Returning_String = self.__str__()
        _Returning_String += (  ( "Size: ( %d, %d )\n" % ( self._Length()[0], self._Length()[1] ) ) )
        return self.__str__()

    def DATA(self):
        return self.__data

    def _Determinant(self):
        return list_determinant(_Array = self.__data)
    def det(self):
        return self._Determinant()

    def _Transposition(self):
        _Array = [ [ self.__data[j][i] for j in range(self._Length()[0]) ] for i in range(self._Length()[1]) ]
        return _Matrix(_Array = _Array)
    def T(self):
        return self._Transposition()

    def _Conjugate(self):
        _Array = [ [ ( self.__data[i][j].conjugate() if type(self.__data[i][j]) == complex else self.__data[i][j] ) for j in range(self._Length()[1]) ] for i in range(self._Length()[0]) ]
        return _Matrix(_Array = _Array)
    def _Conj(self):
        return self._Conjugate()

    def __add__(self, other):
        if self._Length() != other._Length():
            raise Exception
        _Array = [ [ ( self.__data[i][j] + other.__data[i][j] ) for j in range(self._Length()[1]) ] for i in range(self._Length()[0]) ]
        return _Matrix(_Array = _Array)

    def __sub__(self, other):
        if self._Length() != other._Length():
            raise Exception
        _Array = [ [ ( self.__data[i][j] - other.__data[i][j] ) for j in range(self._Length()[1]) ] for i in range(self._Length()[0]) ]
        return _Matrix(_Array = _Array)

    def __mul__(self, other):
        if ( type(self) == _Matrix ) and ( type(other) in NUMERIC_TYPES ):
            _Array = [ [ ( self.__data[i][j] * other ) for j in range(self._Length()[1]) ] for i in range(self._Length()[0]) ]
        elif ( ( type(self) in NUMERIC_TYPES ) and ( type(other) == _Matrix ) ):
            _Array = [ [ ( other.__data[i][j] * self ) for j in range(other._Length()[1]) ] for i in range(other._Length()[0]) ]
        elif ( type(self) == _Matrix ) and ( type(other) == _Matrix ):
            if self._Length()[1] != other._Length()[0]:
                raise Exception("")
            _Array = [ [ sum([ ( self.__data[i][k] * other.__data[k][j] ) for k in range(self._Length()[1]) ]) for j in range(other._Length()[1]) ] for i in range(self._Length()[0]) ]
        else:
            raise Exception("")
        return _Matrix(_Array = _Array)

    def _Print(self):
        for i in range(self._Length()[0]):
            print( end = ( " " * 4 ) )
            for j in range(self._Length()[1]):
                print( self.__data[i][j], end = ", " )
            print()

    def _Vector_Array(self):
        if self._Length()[0] == 1:
            return self.__data[0]
        elif self._Length()[1] == 1:
            return [ self.__data[i][0] for i in range(self._Length()[0]) ]
        else:
            raise Exception("")


def _Vector(_Array = [0], _Type = None):
    if isVec(_Array) == False:
        raise Exception("")
    if _Type == "Ln":
        _ARray = [ [ _Array[i] for i in range(len(_Array)) ] ]
    elif _Type == "Col":
        _ARray = [ [ _Array[i] ] for i in range(len(_Array)) ]
    else:
        raise Exception("")
    return _Matrix(_Array = _ARray)


def _Diag_Matrix(_Array = [0]):
    if isVec(_Array) == False:
        raise Exception("")
    _ARray = [ [ ( _Array[i] if i == j else 0 ) for j in range(len(_Array)) ] for i in range(len(_Array)) ]
    return _Matrix(_Array = _ARray)


def _kE_Matrix( _k = 0, _Diag_Length = 1 ):
    return _Diag_Matrix(_Array = [ _k for i in range(_Diag_Length) ])


def _E_Matrix(_Diag_Length = 1):
    return _kE_Matrix( _k = 1, _Diag_Length = _Diag_Length )


def _kI_Matrix( _k = 0, _Diag_Length = 1 ):
    return _Diag_Matrix(_Array = [ _k for i in range(_Diag_Length) ])


def _I_Matrix(_Diag_Length = 1):
    return _kI_Matrix( _k = 1, _Diag_Length = _Diag_Length )


def _O_Matrix(_Diag_Length = 1):
    return _Matrix(_Array = [ [ 0 for j in range(_Diag_Length) ] for i in range(_Diag_Length) ])


def _Dot( _This, _That ):
    if ( _Matrix in [ type(_This), type(_That) ] ) == False:
        raise Exception("")
    _This, _That = _This._Vector_Array(), _That._Vector_Array()
    if len(_This) != len(_That):
        raise Exception("")
    return sum([ ( _This[i] * _That[i] ) for i in range(len(_This)) ])


list_a = [
    [ 1, 2, 3 ],
    [ 4, 5, 6 ]
    ]
a = _Matrix(list_a)

list_b = [
    [ 0, 1, 0, 0 ],
    [ 0, 0, 1, 0 ],
    [ 0, 0, 0, 1 ],
    [ 0, 0, 0, 0 ]
    ]
b = _Matrix(list_b)

mat1 = [
    [ 1, 0, 0 ],
    [ 0, 1, 0 ],
    [ 0, 0, 1 ]
    ]

mat2 = [
    [ 1, 0, 0 ],
    [ 0, 2, 0 ],
    [ 0, 0, 3 ]
    ]

mat3 = [
    [ 1, 2, 3, 4 ],
    [ 5, 6, 7, 8 ],
    [ 9, 10, 11, 12 ],
    [ 13, 14, 15, 16 ]
    ]
