#!/usr/bin/python
'''
utilities.
'''
from numpy import *
def bcast_dot(A,B):
    '''
    dot product broadcast version.
    '''
    return einsum('...jk,...kl->...jl', A, B)
