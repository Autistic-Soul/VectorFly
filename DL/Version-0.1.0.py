#!/usr/bin/env Python
# -*- coding: utf-8 -*-
import cmath
NUMTYPES = [int, complex, float]

# Activation-Functions

def Sigmoid(x):
    return ( 1 / ( 1 + cmath.exp(-x) ) )

def SinH(x):
    return ( ( cmath.exp(x) - cmath.exp(-x) ) / 2 )

def sh(x):
    return SinH(x)

def CosH(x):
    return ( ( cmath.exp(x) + cmath.exp(-x) ) / 2 )

def ch(x):
    return CosH(x)

def TanH(x):
    return ( SinH(x) / CosH(x) )

def th(x):
    return ( sh(x) / ch(x) )

def ReLU(x):
    return max(0, x)

# Nerual

class Nerual(object):

    def __init__( self, w = [ 0, 0 ] ):
        for each in w:
            if not ( type(each) in NUMTYPES ):
                raise Exception()
        self.W = w

    def compute( self, X = [0], Activation_Function = None ):

        if len(self.W) != ( len(X) + 1 ):
            raise Exception()

        y = self.W[0]
        self.W.pop(0)
        y += sum([ ( X[i] * self.W[i] ) for i in range(len(X)) ])

        if Activation_Function == "SIGMOID":
            return Sigmoid(y)

        elif Activation_Function in [ "TANH", "TH" ]:
            return th(y)

        elif Activation_Function == "RELU":
            return ReLU(y)

        else:
            return y

# Layer

class Layer(object):

    def __init__( self, Nerual_Nodes = [ Nerual() ] ):
        self.nodes = Nerual_Nodes

    def compute( self, X = [0], Activation_Function = None ):
        return [ self.nodes[i].compute( X = X, Activation_Function = Activation_Function ) for i in range(len(self.nodes)) ]

# Net

class Net(object):

    def __init__( self, Layer_Nodes = [ Layer() ] ):
        self.nodes = Layer_Nodes

    def compute( self, X = [0], Activation_Function = None ):
        for i in range(len(self.nodes)):
            print( "X =", X )
            X = self.nodes[i].compute( X = X, Activation_Function = Activation_Function )
        print( "X =", X )
        return X


"""  TESTING PART  """
"""  XOR  PROBLEM  """

n00 = Nerual( w = [ (-0.5), 1, (-1) ] )
n01 = Nerual( w = [ (-0.5), (-1), 1 ] )
n10 = Nerual( w = [ 0.5, 1, 1 ] )

l0 = Layer( Nerual_Nodes = [ n00, n01 ] )
l1 = Layer( Nerual_Nodes = [ n10 ] )

nn = Net( Layer_Nodes = [ l0, l1 ] )

nn.compute(X = [ 1, 0 ])
