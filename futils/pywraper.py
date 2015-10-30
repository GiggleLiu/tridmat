'''
Author: Giggle Leo
Date : 8 September 2014
Description : linear algebra utils library
'''

from numpy import *
from fmodule import *
from numpy.linalg import inv

def ptr2ind(indptr):
    '''
    get x indices of a csr matrix from indptr.

    ptr:
        a indptr of a csr_matrix.
    '''
    return fcsr_xindices(indptr,indptr[-1])

def ind2ptr(inds,N):
    '''
    get indptr of a csr matrix from indices.

    inds:
        the x indices.
    N:
        the matrix size.
    '''
    return fcsr_ptrindices(inds,N+1)



def get_tlu_seq(al,bl,cl,which=None):
    '''
    get the sequences of twisted LU decomposition for tridiagonal block matrix.
    A=LU.
    reference -> http://dx.doi.org/10.1016/j.amc.2005.11.098

    al/bl/cl:
        the lower diagonal/diagonal/upper diagonal part of matrix of length n-1,n,n-1.
    which:
        '<'  -> i<j,
        '>'  -> i>j,
        None -> all of above, together with '='.

    *return*:
        sequences defining L,U
        (ll1,ll1,ll1,ul2,ul2,ul2,hl3,hl3,hl3,invhl1,invhl2,invhl3) if which is None
        (ll,ul,hl,invhl[for block]) if which in `<`,`>`
    '''
    is_scalar=ndim(al)==1
    if which is None:
        return fget_tlu_seq(cl=cl,bl=bl,al=al)
    elif which=='<':
        which=1
    elif which=='>':
        which=2
    else:
        raise ValueError('which is should be `<`,`>`,`=` or None but got %s!'%which)
    if is_scalar:
        ll,ul,hl=fget_tlu_seq1(cl=cl,bl=bl,al=al,which=which)
        return ll,ul,hl,1./hl
    else:
        return fget_tlu_seqn(cl=cl,bl=bl,al=al,which=which)

def get_uv(du,dl,invdu=None,invdl=None):
    '''
    get u,v vectors defining inversion,
    the inversion of A is:
        inv(A) = [u1v1/2, u1v2 ... u1vn] + h.c.
                 [    , u2v2/2 ... u2vn]
                 [         ...         ]
                 [         ...   unvn/2]
    
    du/dl/invdu/invdl:
        the diagonal part of UDL and LDU decomposition and their inversion.
    cl:
        the upper part of tridiagonal matrix.
    
    *return*:
    ul,vl:
        the u,v vectors defining inversion of a matrix.
    '''
    is_scalar=ndim(du)==1
    if is_scalar:
        return fget_uv(dl=dl,du=du)
    else:
        if invdu is None: invdu=inv(du)
        if invdl is None: invdl=inv(du)
        return fget_uvn(dl=dl,du=du,invdu=invdu,invdl=invdl)

def get_dl(al,bl,cl,order):
    '''
    get the diagonal part of UDL decomposition.

    al/bl/cl:
        the lower,diagonal,upper part of tridiagonal matrix
    order:
        `ldu` -> LDU
        `udl` -> UDL

    *return*:
        the diagonal part of ldu/udl decomposition and it's inversion
    '''
    assert(order=='udl' or order=='ldu')
    udl=True if order=='udl' else False
    is_scalar=ndim(al)==1
    if is_scalar:
        dl=fget_dl(al=al,bl=bl,cl=cl,udl=udl)
        return dl,1./dl
    else:
        return fget_dln(al=al,bl=bl,cl=cl,udl=udl)
