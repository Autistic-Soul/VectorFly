#!/usr/bin/env Python
# -*- coding: utf-8 -*-
import math
NUMTYPES = [ int, complex, float ]

def multispaces(length = 4):
    return " " * length

TAB_STR = multispaces()
LINE_BEGIN_STR = "|   "
LINE_END_STR = "\n"

# Activation-Functions

def Linear_Activation_Functions( x, w = 1, b = 0 ):
    return ( w * x + b )

def Identity(x):
    return Linear_Activation_Functions(x)

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
    return max( 0, x )

def PReLU( x, alpha = 0.01 ):
    return max( ( alpha * x ), x )

def Leaky_ReLU(x):
    return PReLU(x)

def ELU( x, alpha = 1 ):
    return max( ( alpha * math.exp(x) - alpha ), x )

# Neural

class Neural(object):

    """
        This is a humble Neural.
        ===
        Descriptions:
            <w>: Weights, the first element is to be the Bias and others are to be the Weights-Vector;
            <Activation_Function>: The type of Activation Function;
        Examples:
            >>> neu = Neural( w = [ 0, 1, (-1) ], Activation_Function = "TANH" )
            >>> neu

            <Neural>:
            |   [Bias]: 0
            |   [Weights] : [1, -1]
    """

    def __init__( self, w = [ 0, 0 ], Activation_Function = None ):

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

    def compute( self, X = [0], Activation_Function = None, Threshold = 0, print_details = False ):

        if len(self.W) != ( len(X) + 1 ):
            raise Exception()

        X.reverse()
        X.append(1)
        X.reverse()
        y = sum([ ( self.W[i] * X[i] ) for i in range(len(self.W)) ])

        if Activation_Function == "LINEAR_THRESHOLD":
            y_pred =  Linear_Threshold( y, threshold = Threshold )
        elif Activation_Function in [ "SIGMOID", "SIG" ]:
            y_pred =  Sigmoid(y)
        elif Activation_Function in [ "TANH", "TH" ]:
            y_pred =  th(y)
        elif Activation_Function == "RELU":
            y_pred = ReLU(y)
        else:
            y_pred = y

        if print_details:
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, "<Neural Computing...>:", LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, ( "X_Input = %s" % X[ 1 : ] ), LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, ( "W = %s" % self.W[ 1 : ] ), LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, ( "b = %s" % self.W[0] ), LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, "y =", LINE_END_STR, sep = "", end = "" )
            for i in range( 1, len(self.W) ):
                print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, TAB_STR, sep = "", end = "" )
                if ( i > 1 ) and ( self.W[i] >= 0 ):
                    print("+", sep = "", end = "")
                print( ( "%s * %s" % ( self.W[i], X[i] ) ), sep = "", end = "" )
                print( LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, TAB_STR, ( "%s" % ( self.W[0] ) ), LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, "  = ", y, LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, ( "Activation_Function: %s" % Activation_Function ), LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, ( "y_Output = %s" % y_pred ), LINE_END_STR, sep = "", end = "" )

        X.pop(0)

        return y_pred

# Layer

class Layer(object):

    """
        This is a humble Layer.
        ===
        Descriptions:
            <Neural_Nodes>: The Neurals which is in this Layer;
    """

    def __init__( self, Neural_Nodes = [ Neural() ] ):
        self.nodes = Neural_Nodes

    def __repr__(self):
        pass

    def compute( self, X = [0], Activation_Function = None, Threshold = 0, print_details = False ):
        if print_details:
            print( LINE_BEGIN_STR, "<Layer Computing...>:", LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, ( "X = %s" % X ), LINE_END_STR, sep = "", end = "" )
        y = [ self.nodes[i].compute( X = X, Activation_Function = Activation_Function, Threshold = Threshold, print_details = print_details ) for i in range(len(self.nodes)) ]
        if print_details:
            print( LINE_BEGIN_STR, ( "y = %s" % y ), LINE_END_STR, sep = "", end = "" )
        return y

# Net

class Net(object):

    """
        This is a humble Net.
        ===
        Descriptions:
            <Layer_Nodes>: The Layers which is in this Net;
    """

    def __init__( self, Layer_Nodes = [ Layer() ] ):
        self.nodes = Layer_Nodes

    def compute( self, X = [0], Activation_Function = None, Threshold = 0, print_details = False, print_results = True ):
        print( ( "Inputting_X: %s" % X ), LINE_END_STR, sep = "", end = "" )
        for i in range(len(self.nodes)):
            X = self.nodes[i].compute( X = X, Activation_Function = Activation_Function, Threshold = Threshold, print_details = print_details )
        if print_results:
            print( "Outputting_y =", X )
        return X


""" [TESTING-PART]
    Descriptions:
        input_X     ||      output_y
        ( 0, 0 )    ->      0
        ( 0, 1 )    ->      1
        ( 1, 0 )    ->      1
        ( 1, 1 )    ->      0
    Standard Answer:
        net : {
            layer : {
                neural( weights = [ 1, -1 ], bias = -0.5 ),
                neural( weights = [ -1, 1 ], bias = -0.5 )
                },
            layer : {
                neural( weights = [ 1, 1 ], bias = 0.5 )
                }
            }
"""

X_TEST = [
    [ 0, 0 ],
    [ 0, 1 ],
    [ 1, 0 ],
    [ 1, 1 ]
    ]

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
for i in range(len(X_TEST)):
    print( "=" * 20 )
    net_1303.compute( X = X_TEST[i], Activation_Function = "RELU", print_details = True )
"""
net_1303.compute( X = X_TEST[-1], Activation_Function = "RELU", print_details = True )
"""

