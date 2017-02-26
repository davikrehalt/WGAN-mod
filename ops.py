from __future__ import print_function,division
import numpy as np
import theano
import theano.tensor as T

def tmax(input,bar):
    return T.switch(input>bar,input,bar)
def tmin(input,bar):
    return T.switch(input<bar,input,bar)
def tbox(input,low,high):
    return tmin(tmax(input,low),high)
