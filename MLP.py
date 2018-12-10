#!/usr/bin/env Python
# -*- coding: utf-8 -*-
################################################################################



################################################################################
# PHASE1 started on Oct 11th, 13:11:26, 2018

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

def softplus(x):
    return math.log( math.exp(x) + 1 )

def sinc(x):
    return ( math.sin(x) / x ) if x != 0 else 1

def Gaussian_Activation(x):
    return math.exp( -( x ** 2 ) )

# Neuron

class Neuron(object):

    """
        This is a humble Neuron.
        ===
        Descriptions:
            <w>: Weights, the first element is to be the Bias and others are to be the Weights-Vector;
            <Activation>: The type of Activation Function;
        Examples:
            >>> neu = Neuron( w = [ 0.5, 1, 1 ], activation = "TANH" )
            >>> neu

            <Neuron>:
            |   [Bias]: 0.5
            |   [Weights]: [1, 1]
            |   [Activation]: TANH
    """

    def __init__( self, w = [ 0, 0 ], activation = None ):
        for each in w:
            if not ( type(each) in NUMTYPES ):
                raise Exception()
        self.W = w
        self.Activation = activation if activation != None else None

    def __repr__(self):

        returning_string = LINE_END_STR
        returning_string += "<Neuron>:"
        returning_string += LINE_END_STR
        returning_string += LINE_BEGIN_STR
        returning_string += "[Bias]: %s" % self.W[0]
        returning_string += LINE_END_STR
        returning_string += LINE_BEGIN_STR
        returning_string += "[Weights]: %s" % self.W[ 1 : ]
        returning_string += LINE_END_STR
        returning_string += LINE_BEGIN_STR
        returning_string += "[Activation]: %s" % self.Activation
        returning_string += LINE_END_STR

        return returning_string

    def compute( self, X = [0], activation = None, Threshold = 0, print_details = False ):

        if len(self.W) != ( len(X) + 1 ):
            raise Exception()

        X.reverse()
        X.append(1)
        X.reverse()
        y = sum([ ( self.W[i] * X[i] ) for i in range(len(self.W)) ])

        if activation == "LINEAR_THRESHOLD":
            y_pred =  Linear_Threshold( y, threshold = Threshold )
        elif activation in [ "SIGMOID", "SIG" ]:
            y_pred =  Sigmoid(y)
        elif activation in [ "TANH", "TH" ]:
            y_pred =  th(y)
        elif activation == "RELU":
            y_pred = ReLU(y)
        else:
            y_pred = y

        if print_details:
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, "<Neuron Computing> Begin:", LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, ( "X_Input = %s" % X[ 1 : ] ), LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, ( "W = %s" % self.W[ 1 : ] ), LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, ( "b = %s" % self.W[0] ), LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, ( "y = %s" % self.W[0] ), LINE_END_STR, sep = "", end = "" )
            for i in range( 1, len(self.W) ):
                print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, TAB_STR, sep = "", end = "" )
                if ( i > 1 ) and ( self.W[i] >= 0 ):
                    print("+ ", sep = "", end = "")
                print( ( "%s * %s" % ( self.W[i], X[i] ) ), sep = "", end = "" )
                print( LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, "  = ", y, LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, ( "Activation_Function: %s" % activation ), LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, LINE_BEGIN_STR, ( "y_Output = %s" % y_pred ), LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, "<Neuron Computing> End.", LINE_END_STR, sep = "", end = "" )

        X.pop(0)

        return y_pred

# Layer

class Layer(object):

    """
        This is a humble Layer.
        ===
        Descriptions:
            <Neuron_Nodes>: The Neurons which is in this Layer;
    """

    def __init__( self, Neuron_Nodes = [ Neuron() ] ):
        self.nodes = Neuron_Nodes

    def __repr__(self):

        returning_string = LINE_END_STR
        returning_string += "(Layer):" + LINE_END_STR
        for each_neural in self.nodes:
            returning_string += LINE_BEGIN_STR + "<Neuron>:" + LINE_END_STR
            returning_string += LINE_BEGIN_STR * 2 + "[Bias]: %s" % each_neural.W[0] + LINE_END_STR
            returning_string += LINE_BEGIN_STR * 2 + "[Weights]: %s" % each_neural.W[ 1 : ] + LINE_END_STR
            returning_string += LINE_BEGIN_STR * 2 + "[Activation]: %s" % each_neural.Activation + LINE_END_STR

        return returning_string

    def compute( self, X = [0], activation = None, Threshold = 0, print_details = False ):
        if print_details:
            print( LINE_BEGIN_STR, "<Layer Computing> Begin:", LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, ( "X = %s" % X ), LINE_END_STR, sep = "", end = "" )
        y = [ self.nodes[i].compute( X = X, activation = activation, Threshold = Threshold, print_details = print_details ) for i in range(len(self.nodes)) ]
        if print_details:
            print( LINE_BEGIN_STR, LINE_BEGIN_STR, ( "y = %s" % y ), LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, "<Layer Computing> End.", LINE_END_STR, sep = "", end = "" )
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

    def compute( self, X = [0], activation = None, Threshold = 0, print_details = False ):
        if print_details:
            print( "<Net Computing> Begin:", LINE_END_STR, sep = "", end = "" )
            print( LINE_BEGIN_STR, ( "Inputting_X: %s" % X ), LINE_END_STR, sep = "", end = "" )
        for i in range(len(self.nodes)):
            X = self.nodes[i].compute( X = X, activation = activation, Threshold = Threshold, print_details = print_details )
        if print_details:
            print( LINE_BEGIN_STR, "Outputting_y =", X )
            print( "<Net Computing> End." )
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

net_1303 = Net(
    Layer_Nodes = [
        Layer(
            Neuron_Nodes = [ Neuron( w = [ 0, 1, (-1) ] ), Neuron( w = [ 0, (-1), 1 ] ) ]
            ),
        Layer(
            Neuron_Nodes = [ Neuron( w = [ 0, 1, 1 ] ) ]
            )
        ]
    )
for i in range(len(X_TEST)):
    print( "=" * 20 )
    net_1303.compute( X = X_TEST[i], activation = "RELU", print_details = True )

neural_1252 = Neuron( w = [ 0.5, 1, 1 ], activation = "TANH" )

layer_1315 = Layer( [ Neuron( [ -0.5, 1, -1 ], activation = "TANH" ), Neuron( [ -0.5, -1, 1 ], activation = "TANH" ) ] )

################################################################################



################################################################################
# PHASE2 Started on Oct 29th, 16:08:45, 2018
# Based on [theano]
import numpy as np
import theano
from theano import tensor as T

class NeuronV2(object):

    """
        A Humble Neuron by czk.
        ===
    """

    def __init__( self, Weights = [0], Bias = 0, Activation_Function = None ):
        self.__Weights = Weights
        self.__Bias = Bias
        self.__Activation = Activation_Function
        return

    def __add__(self, other):
        if len(self.__Weights) != len(other.__Weights) or self.__Activation != other.__Activation:
            raise Exception
        return NeuronV20(
            Weights = [ ( self.__Weights[i] + other.__Weights[i] ) for i in range(len(self.__Weights)) ],
            Bias = ( self.__Bias + other.__Bias ),
            Activation_Function = self.__Activation
            )

    def __sub__(self, other):
        if len(self.__Weights) != len(other.__Weights) or self.__Activation != other.__Activation:
            raise Exception
        return NeuronV20(
            Weights = [ ( self.__Weights[i] - other.__Weights[i] ) for i in range(len(self.__Weights)) ],
            Bias = ( self.__Bias - other.__Bias ),
            Activation_Function = self.__Activation
            )

    def compute( self, X = None, print_details = False ):
        if len(X) != len(self.__Weights):
            raise Exception()
        y = np.dot( a = self.__Weights, b = X ) + self.__Bias
        return self.__Activation(y) if self.__Activation != None else y
