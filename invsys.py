#!/usr/bin/python
'''
quick inversion for specific element with tridiagonalized matrix.

In the following description, we take p -> the block dimension, N -> the matrix dimension and n = N/p.

*references*:
    * http://dx.doi.org/10.1137/0613045
    * http://dx.doi.org/10.1016/j.amc.2005.11.098
'''
from numpy import *
from scipy.sparse import bmat as sbmat
import pdb,time

class InvSystem(object):
    '''
    The abstract interface of inversion generator for tridiagonal matrix.
    '''
    def get_item(self,i,j,*args,**kwargs):
        '''
        Get specific item of the inverse matrix.

        i/j:
            The row/column index.
        '''
        raise Exception('Not Implemented')

    def get_all(self,*args,**kwargs):
        '''
        Get the inverse of matrix.
        '''
        raise Exception('Not Implemented')

class STInvSystem(InvSystem):
    '''
    Matrix Inversion Fast Generator for Scalar Tridiagonal Matrix.

    ul,vl:
        The u,v vectors defining this inversion.
    '''
    def __init__(self,ul,vl):
        self.ul=ul
        self.vl=vl

    @property
    def n(self):
        '''The number of blocks.'''
        return len(self.ul)

    def get_item(self,i,j):
        '''
        Get specific item of the inverse matrix.

        i/j:
            The row/column index.
        *return*:
            A number.
        '''
        if i<=j:
            return self.ul[i]*vl[j]
        else:
            return self.ul[j]*vl[i]

    def get_all(self):
        '''
        Get the inverse of matrix.

        *return*:
            A (N x N) array.
        '''
        m=self.ul[:,newaxis].dot(self.vl[newaxis,:])
        m=triu(m)+triu(m,1).T.conj()
        return m


class BTInvSystem(InvSystem):
    '''
    Fast Generator for the inversion of A Tridiagonal Matrix.

    seq_lt/seq_gt/seq_eq:
        sequence for i < j/i > j/i==j
        sequences are aranged in the order
    '''
    def __init__(self,seq_lt,seq_gt,seq_eq):
        self.ll_lt,self.hl_lt,self.ul_lt,self.invhl_lt=seq_lt
        self.ll_gt,self.hl_gt,self.ul_gt,self.invhl_gt=seq_gt
        self.ll_eq,self.hl_eq,self.ul_eq,self.invhl_eq=seq_eq

    @property
    def is_scalar(self):
        '''True if functionning as scalar.'''
        return ndim(self.hl_lt)==1

    def __get_L__(self,i,j):
        '''Get L(i,j)'''
        if i<j:
            return self.ll_lt[i]
        elif i>j:
            return self.ll_gt[i]
        else:
            return self.ll_eq[i]

    def __get_H__(self,i,j):
        '''Get H(i,j)'''
        if i<j:
            return self.hl_lt[i]
        elif i>j:
            return self.hl_gt[i]
        else:
            return self.hl_eq[i]

    def __get_invH__(self,i,j):
        '''Get H(i,j)^{-1}'''
        if i<j:
            return self.invhl_lt[i]
        elif i>j:
            return self.invhl_gt[i]
        else:
            return self.invhl_eq[i]

    def __get_U__(self,i,j):
        '''Get U(i,j)'''
        if i<j:
            return self.ul_lt[i]
        elif i>j:
            return self.ul_gt[i]
        else:
            return self.ul_eq[i]

    @property
    def n(self):
        '''The number of blocks.'''
        return len(self.hl_eq)

    @property
    def p(self):
        '''The block size'''
        if self.is_scalar:
            return 1
        else:
            return shape(self.hl_eq)[-1]

    def get_twist_LU(self,j):
        '''
        Get the twist LU decomposition of the original matrix.

        For the definnition of twist LU decomposition, check the references of this module.

        j:
            The twisting position.

        *return*:
            (L,U), each of which is a (N x N) sparse matrix.
        '''
        n=self.n
        p=self.p
        L=ndarray((n,n),dtype='O')
        U=ndarray((n,n),dtype='O')
        I=identity(p)
        for i in xrange(n):
            L[i,i]=self.__get_H__(i,j)
            U[i,i]=I
            if i<j:
                L[i+1,i]=self.__get_L__(i,j)
                U[i,i+1]=self.__get_U__(i,j)
            elif i>j:
                L[i-1,i]=self.__get_L__(i-1,j-1)
                U[i,i-1]=self.__get_U__(i-1,j-1)
        L=sbmat(L)
        U=sbmat(U)
        return L,U

    def get_item(self,i,j):
        '''
        Get specific item of the inverse matrix.

        i/j:
            The row/column index.

        *return*:
            Matrix element, A (p x p) array for block version or a number for scalar version.
        '''
        if i==j:
            return self.__get_invH__(i,j)
        elif i<j:
            return dot(-self.__get_U__(i,j),self.__get_item__(i+1,j))
        else:
            return dot(-self.__get_U__(i-1,j),self.__get_item__(i-1,j))

    def get_col(self,j):
        '''
        Get the specific column of the inverse matrix.

        j:
            The column index.

        *return*:
            The specific column of the inversion of tridiagonal matrix,
            An array of shape (n*p x p) for block version or (n) for scalar version.
        '''
        cjj=self.get_item(j,j)
        cl=[cjj]
        ci=cjj
        n=len(self.hl_eq)
        for i in xrange(j+1,n):
            ci=dot(-self.__get_U__(i-1,j),ci)
            cl.append(ci)
        ci=cjj
        for i in xrange(j-1,-1,-1):
            ci=dot(-self.__get_U__(i,j),ci)
            cl.insert(0,ci)
        return cl

    def get_all(self):
        '''
        Get the inverse of matrix.

        *return*:
            A (N x N) array.
        '''
        n,p=self.n,self.p
        m=[]
        for j in xrange(n):
            m.append(self.get_col(j))
        if self.is_scalar:
            m=transpose(m)
        else:
            m=transpose(m,axes=(1,2,0,3)).reshape([n*p,n*p])
        return m


