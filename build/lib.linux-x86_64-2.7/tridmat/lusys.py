'''
The LU Decomposition system for tridiagonal matrix.

In the following description, we use the convention that p -> the block dimension, N -> the matrix dimension and n = N/p.
'''

from numpy import *
from numpy.linalg import inv
from scipy.sparse import bmat as sbmat
from scipy.sparse import bsr_matrix,csr_matrix,block_diag

from futils.pywraper import ind2ptr
from trid import build_sr

__all__=['TLUSystem']

class TLUSystem(object):
    '''
    LU/UL/LDU/UDL decomposition matrix generator class for a general tridiagonal matrix.

    ll/ul/hl/invhl:
        The lower part of L/the upper part of U/the diagonal part of left matrix and it's inverse.
    order:
        'LU'/'UL'/'LDU' or 'UDL', defining the multiplying order to recover the original matrix.
    '''
    def __init__(self,ll,ul,hl,invhl,order):
        assert(order in ['LU','UL','LDU','UDL'])
        self.ll=ll
        self.hl=hl
        self.ul=ul
        self.invhl=invhl
        self.order=order

    @property
    def n(self):
        '''Size with respect to block'''
        return len(self.hl)

    @property
    def p(self):
        '''Block size.'''
        return 1 if self.is_scalar else self.hl.shape[-1]

    @property
    def is_scalar(self):
        '''True if it is functionning as a scalar version.'''
        return ndim(self.hl)==1

    @property
    def L(self):
        '''
        The the lower triangular matrix.
        '''
        if self.order=='UL':
            if self.is_scalar:
                dl=ones(self.n)
            else:
                dl=[identity(self.p)]*n
        elif self.order=='LU' or self.order=='UDL' or self.order=='LDU':
            dl=self.hl
        L=build_sr(dl=dl,ll=self.ll)
        return L

    @property
    def U(self):
        '''
        The the upper triangular matrix.
        '''
        if self.order=='LU':
            if self.is_scalar:
                dl=ones(self.n)
            else:
                dl=[identity(self.p)]*self.n
        elif self.order=='UL' or self.order=='UDL' or self.order=='LDU':
            dl=self.hl
        U=build_sr(dl=dl,ul=self.ul)
        return U

    @property
    def D(self):
        '''
        The diagonal matrix in LDU or UDL decomposition.
        '''
        if self.order=='LU' or self.order=='UL':
            return None
        if self.invhl is None:
            if self.is_scalar:
                self.invhl=1./self.hl
            else:
                self.invhl=inv(self.hl)
        return block_diag(self.invhl)
