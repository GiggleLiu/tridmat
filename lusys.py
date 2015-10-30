#!/usr/bin/python
'''
The LU Decomposition system for tridiagonal matrix.
'''
from numpy import *
from numpy.linalg import inv
from scipy.sparse import bmat as sbmat
from scipy.sparse import bsr_matrix,csr_matrix,block_diag
from futils.pywraper import ind2ptr
import pdb,time

class TLUSystem(object):
    '''
    lu/ul decomposition matrix generator class.

    ll/ul/hl/invhl:
        the lower part of L, the upper part of U, the diagonal part of left matrix and it's inverse.
    order:
        `LU` or `UL`
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
        '''size with respect to block'''
        return len(self.hl)

    @property
    def p(self):
        '''block size.'''
        return 1 if self.is_scalar else self.hl.shape[-1]

    @property
    def is_scalar(self):
        '''return True if is scalar.'''
        return ndim(self.hl)==1

    @property
    def L(self):
        '''
        The the lower triangular matrix.
        '''
        n=self.n
        p=self.p
        ll=self.ll
        is_scalar=self.is_scalar
        if is_scalar:
            mgen=csr_matrix
        else:
            mgen=bsr_matrix
        indx_d=arange(n)
        indx_l=concatenate([indx_d,indx_d[1:]])
        indy_l=concatenate([indx_d,indx_d[:-1]])
        if self.order=='UL':
            if is_scalar:
                data_l=concatenate([ones(n),ll])
            else:
                data_l=concatenate([[identity(p)]*n,ll])
        elif self.order=='LU' or self.order=='UDL' or self.order=='LDU':
            data_l=concatenate([self.hl,ll])
        odl=argsort(indx_l)
        L=mgen((data_l[odl],indy_l[odl],ind2ptr(indx_l[odl],n)),dtype=ll.dtype)
        return L

    @property
    def U(self):
        '''
        The the upper triangular matrix.
        '''
        n=self.n
        p=self.p
        is_scalar=self.is_scalar
        ul=self.ul
        if is_scalar:
            mgen=csr_matrix
        else:
            mgen=bsr_matrix
        indx_d=arange(n)
        indx_u=concatenate([indx_d,indx_d[:-1]])
        indy_u=concatenate([indx_d,indx_d[1:]])
        if self.order=='LU':
            if is_scalar:
                data_u=concatenate([ones(n),ul])
            else:
                data_u=concatenate([[identity(p)]*n,ul])
        elif self.order=='UL' or self.order=='UDL' or self.order=='LDU':
            data_u=concatenate([self.hl,ul])
        odu=argsort(indx_u)
        U=mgen((data_u[odu],indy_u[odu],ind2ptr(indx_u[odu],n)),dtype=ul.dtype)
        return U

    @property
    def D(self):
        '''
        The diagonal part of LDU or UDL decomposition.
        '''
        if self.order=='LU' or self.order=='UL':
            return None
        if self.invhl is None:
            if self.is_scalar:
                self.invhl=1./self.hl
            else:
                self.invhl=inv(self.hl)
        return block_diag(self.invhl)
