'''
Linear algebra for block and tridiagonal matrices.

In the following description, we take p -> the block dimension, N -> the matrix dimension and n = N/p.
'''

from numpy.linalg import inv
from numpy import *
from scipy.sparse import coo_matrix,bsr_matrix,csr_matrix,block_diag
from scipy.sparse import bmat as sbmat

from invsys import STInvSystem,BTInvSystem
from lusys import TLUSystem
from futils.pywraper import get_tlu_seq,ind2ptr,get_dl,get_uv

__all__=['trilu','triul','trildu','triudl','get_invh_system','get_inv_system','tinv']

def trilu(tridmat):
    '''
    Get the LU decomposition for tridiagonal matrix, A=LU

    The complexity of this procedure is n*p^3.

    trdmat:
        A tridiagonal matrix, i.g. an instance of TridMatrix.

    *return*:
        A <TLUSystem> instance with order 'LU'.
    '''
    al=tridmat.lower
    bl=tridmat.diagonal
    cl=tridmat.upper
    is_scalar=tridmat.is_scalar
    ll,ul,hl,invhl=get_tlu_seq(al=al,bl=bl,cl=cl,which='<')
    return TLUSystem(ll=ll,ul=ul,hl=hl,invhl=invhl,order='LU')

def triul(tridmat):
    '''
    Get the UL decomposition for tridiagonal matrix, A=UL

    The complexity of this procedure is n*p^3.

    *return*:
        A <TLUSystem> instance with order 'UL'.
    '''
    al=tridmat.lower
    bl=tridmat.diagonal
    cl=tridmat.upper
    ll,ul,hl,invhl=get_tlu_seq(al=al,bl=bl,cl=cl,which='>')
    return TLUSystem(ll=ul,ul=ll,hl=hl,invhl=invhl,order='UL')

def trildu(tridmat):
    '''
    Get the LDU decomposition, A=LD^{-1}L^\dag

    The complexity of this procedure is n*p^3.

    trdmat:
        A tridiagonal matrix, i.g. an instance of TridMatrix.

    *return*:
        A <TLUSystem> instance with order 'LDU'.
    '''
    dl,invdl=get_dl(al=tridmat.lower,bl=tridmat.diagonal,cl=tridmat.upper,order='LDU')
    res=TLUSystem(hl=dl,invhl=invdl,ll=tridmat.lower,ul=tridmat.upper,order='LDU')
    return res

def triudl(tridmat):
    '''
    Get the UDL decomposition, A=UD^{-1}U^\dag

    The complexity of this procedure is n*p^3.

    tridmat:
        The tridiagonal matrix.

    *return*:
        A <TLUSystem> instance with order 'UDL'.
    '''
    dl,invdl=get_dl(al=tridmat.lower,bl=tridmat.diagonal,cl=tridmat.upper,order='UDL')
    return TLUSystem(hl=dl,invhl=invdl,ll=tridmat.lower,ul=tridmat.upper,order='UDL')

def get_invh_system(tridmat):
    '''
    Get the inversion generator for `hermitian` tridiagonal matrix.
    The Fortran version and python version are provided for block tridiagonal matrix.

    The complexity of this procedure is N.
    However, if you're going to generate the whole inversion elements through STInvSystem instance,
    the complexity is N^2.

    Reference -> http://dx.doi.org/10.1137/0613045

    trdmat:
        A tridiagonal matrix, i.g. an instance of TridMatrix.

    *return*:
        A STInvSystem instance.
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

    The complexity of this procedure is n*p^3.
    However, if you're going to generate the whole inversion elements through STInvSystem instance,
    the complexity is n^2*p^3.

    Reference -> http://dx.doi.org/10.1016/j.amc.2005.11.098

    trdmat:
        A tridiagonal matrix, i.g. an instance of TridMatrix.

    *return*:
        a <BTInvSystem> instance.
    '''
    ll1,ll2,ll3,ul1,ul2,ul3,hl1,hl2,hl3,invhl1,invhl2,invhl3=get_tlu_seq(al=tridmat.lower,bl=tridmat.diagonal,cl=tridmat.upper)
    return BTInvSystem((ll1,hl1,ul1,invhl1),(ll2,hl2,ul2,invhl2),(ll3,hl3,ul3,invhl3))

def tinv(tridmat):
    '''
    Get the inversion of a tridiagonal matrix.

    trdmat:
        A tridiagonal matrix, i.g. an instance of TridMatrix.

    *return*:
        A two dimensional array, the inversion matrix of tridmat.
    '''
    invs=get_inv_system(tridmat)
    return invs.toarray()


