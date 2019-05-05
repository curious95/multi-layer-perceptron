import numpy as np
import math

def tanH_act(val):
    num = math.exp(val)-math.exp(-val)
    den = math.exp(val)+math.exp(-val)
    return num/den;

def sigmoid_act(val):
    num = 1
    den = 1+math.exp(-val)
    return num/den;


