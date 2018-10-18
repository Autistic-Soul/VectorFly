#!/usr/bin/env Python
# -*- coding: utf-8 -*-
import math
NUMTYPES = [int, complex, float]

def multispaces(length = 4):
    return " " * length

TAB_STR = multispaces()
LINE_BEGIN_STR = "|   "
LINE_END_STR = "\n"

X_TEST = [ 1, 1 ]

"""
    input_X     ||      output_y
    ( 0, 0 )    ->      0
    ( 0, 1 )    ->      1
    ( 1, 0 )    ->      1
    ( 1, 1 )    ->      0
"""

# Activation-Functions

def Linear_Threshold(x, threshold = 0):
    return 1 if ( x >= threshold ) else 0

def Sigmoid(x):
    return ( 1 / ( 1 + math.exp(-x) ) )

def SinH(x):
    return ( ( math.exp(x) - math.exp(-x) ) / 2 )

def sh(x):
    return SinH(x)

def CosH(x):
    return ( ( math.exp(x) + math.exp(-x) ) / 2 )

def ch(x):
    return CosH(x)

def TanH(x):
    return ( SinH(x) / CosH(x) )

def th(x):
    return ( sh(x) / ch(x) )

def ReLU(x):
    return max(0, x)

# Neural

class Neural(object):

    def __init__( self, w = [ 0, 0 ] ):

        for each in w:
            if not ( type(each) in NUMTYPES ):
                raise Exception()

        self.W = w

    def __repr__(self):

        returning_string = LINE_END_STR
        returning_string += "<Neural>:"
        returning_string += LINE_END_STR
        returning_string += LINE_BEGIN_STR
        returning_string += "[Bias]: %s" % self.W[0]
        returning_string += LINE_END_STR
        returning_string += LINE_BEGIN_STR
        returning_string += "[Weights] : %s" % self.W[ 1 : ]
        returning_string += LINE_END_STR

        return returning_string

    def compute( self, X = [0], Activation_Function = None, Threshold = 0 ):

        if len(self.W) != ( len(X) + 1 ):
            raise Exception()

        print( "<Computing...>:", LINE_END_STR, sep = "", end = "" )
        print( LINE_BEGIN_STR, ( "X = %s" % X ), LINE_END_STR, sep = "", end = "" )
        X.reverse()
        X.append(1)
        X.reverse()
        print( LINE_BEGIN_STR, ( "W = %s" % self.W[ 1 : ] ), LINE_END_STR, sep = "", end = "" )
        print( LINE_BEGIN_STR, ( "b = %s" % self.W[0] ), LINE_END_STR, sep = "", end = "" )
        print( LINE_BEGIN_STR, "y =", LINE_END_STR, sep = "", end = "" )
        for i in range( 1, len(self.W) ):
            print( LINE_BEGIN_STR, TAB_STR, sep = "", end = "" )
            if ( i > 1 ) and ( self.W[i] >= 0 ):
                print("+", sep = "", end = "")
            print( ( "%s * %s" % ( self.W[i], X[i] ) ), sep = "", end = "" )
            print( LINE_END_STR, sep = "", end = "" )
        print( LINE_BEGIN_STR, TAB_STR, ( "%s" % ( self.W[0] ) ), LINE_END_STR, sep = "", end = "" )
        y = sum([ ( self.W[i] * X[i] ) for i in range(len(self.W)) ])
        print( LINE_BEGIN_STR, "  = ", y, LINE_END_STR, sep = "", end = "" )

        X.pop(0)

        if Activation_Function == "LINEAR_THRESHOLD":
            return Linear_Threshold( y, threshold = Threshold )
        elif Activation_Function in [ "SIGMOID", "SIG" ]:
            return Sigmoid(y)
        elif Activation_Function in [ "TANH", "TH" ]:
            return th(y)
        elif Activation_Function == "RELU":
            return ReLU(y)
        else:
            return y

# Layer

class Layer(object):

    def __init__( self, Neural_Nodes = [ Neural() ] ):
        self.nodes = Neural_Nodes

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
        print( "y =", X )
        return X


"""TESTING-PART"""

#1
net_1303 = Net(
    Layer_Nodes = [
        Layer(
            Neural_Nodes = [ Neural( w = [ 0, 1, (-1) ] ), Neural( w = [ 0, (-1), 1 ] ) ]
            ),
        Layer(
            Neural_Nodes = [ Neural( w = [ 0, 1, 1 ] ) ]
            )
        ]
    )
net_1303.compute( X = X_TEST, Activation_Function = "RELU" )

#2
net_1311 = Net(
    Layer_Nodes = [
        Layer(),
        Layer()
        ]
    )
