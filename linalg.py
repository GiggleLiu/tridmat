#!/usr/bin/python
'''
linear algebra for block and tridiagonal matrices.
'''
from numpy.linalg import inv
from numpy import *
from invsys import STInvSystem,BTInvSystem
from lusys import TLUSystem
from futils.pywraper import get_tlu_seq,ind2ptr,get_dl,get_uv
from scipy.sparse import coo_matrix,bsr_matrix,csr_matrix,block_diag
from scipy.sparse import bmat as sbmat
import pylinalg as pylin
import pdb,time

def trilu(tridmat):
    '''
    get the LU decomposition for tridiagonal matrix, A=LU

    *return*:
        (L,U), lower diagonal and upper diagonal matrices.
    '''
    al=tridmat.lower
    bl=tridmat.diagonal
    cl=tridmat.upper
    is_scalar=tridmat.is_scalar
    ll,ul,hl,invhl=get_tlu_seq(al=al,bl=bl,cl=cl,which='<')
    return TLUSystem(ll=ll,ul=ul,hl=hl,invhl=invhl,order='LU')

def triul(tridmat):
    '''
    get the UL decomposition for tridiagonal matrix, A=UL

    *return*:
        (U,L), upper diagonal and lower diagonal matrices.
    '''
    al=tridmat.lower
    bl=tridmat.diagonal
    cl=tridmat.upper
    ll,ul,hl,invhl=get_tlu_seq(al=al,bl=bl,cl=cl,which='>')
    return TLUSystem(ll=ul,ul=ll,hl=hl,invhl=invhl,order='UL')

def trildu(tridmat):
    '''
    get the LDU decomposition, A=LD^{-1}L^\dag

    tridmat:
        the tridiagonal matrix.

    *return*:
        L = diag(dl)+lowerdiag(tridmat.lower)
        D = diag(dl)
    '''
    dl,invdl=get_dl(al=tridmat.lower,bl=tridmat.diagonal,cl=tridmat.upper,order='LDU')
    res=TLUSystem(hl=dl,invhl=invdl,ll=tridmat.lower,ul=tridmat.upper,order='LDU')
    return res

def triudl(tridmat):
    '''
    get the UDL decomposition, A=UD^{-1}U^\dag

    tridmat:
        the tridiagonal matrix.

    *return*:
        U = diag(dl)+upperdiag(tridmat.upper)
        D = diag(dl)
    '''
    dl,invdl=get_dl(al=tridmat.lower,bl=tridmat.diagonal,cl=tridmat.upper,order='UDL')
    return TLUSystem(hl=dl,invhl=invdl,ll=tridmat.lower,ul=tridmat.upper,order='UDL')

def get_invh_system(tridmat):
    '''
    Get the inversion generator for `hermitian` tridiagonal matrix.
    The Fortran version and python version are provided for block tridiagonal matrix.

    reference -> http://dx.doi.org/10.1016/j.amc.2005.11.098

    py:
        python version prefered if True.

    *return*:
        Inv system instance.
    '''
    if tridmat.is_scalar:
        du,invdu=get_dl(al=tridmat.lower,bl=tridmat.diagonal,cl=tridmat.upper,order='UDL')
        dl,invdl=get_dl(al=tridmat.lower,bl=tridmat.diagonal,cl=tridmat.upper,order='LDU')
        u,v=get_uv(invdu=invdu,invdl=invdl,al=tridmat.lower,cl=tridmat.upper)
        res=STInvSystem(u,v)
        return res
    else:
        raise Exception('Not Implemented For block tridiagonal matrices!')
 
def get_inv_system(tridmat):
    '''
    Get the inversion generator for tridiagonal matrix.
    The Fortran version and python version are provided for block tridiagonal matrix.

    reference -> http://dx.doi.org/10.1016/j.amc.2005.11.098

    py:
        python version prefered if True.

    *return*:
        Inv system instance.
    '''
    ll1,ll2,ll3,ul1,ul2,ul3,hl1,hl2,hl3,invhl1,invhl2,invhl3=get_tlu_seq(al=tridmat.lower,bl=tridmat.diagonal,cl=tridmat.upper)
    return BTInvSystem((ll1,hl1,ul1,invhl1),(ll2,hl2,ul2,invhl2),(ll3,hl3,ul3,invhl3))

def tinv(tridmat):
    '''
    get the inversion of this matrix.
    '''
    invs=get_inv_system(tridmat)
    return invs.get_all()


