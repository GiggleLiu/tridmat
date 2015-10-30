#!/usr/bin/python
'''
linear algebra for block and tridiagonal matrices.
'''
from numpy.linalg import inv
from numpy import *
from invsys import STInvSystem,BTInvSystem
from lusys import TLUSystem
from futils.pywraper import get_tlu_seq,ind2ptr,get_dl
from scipy.sparse import coo_matrix,bsr_matrix,csr_matrix,block_diag
from scipy.sparse import bmat as sbmat
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

def get_inv_system(tridmat,py=False):
    '''
    Get the inversion generator for tridiagonal matrix.
    The Fortran version and python version are provided for block tridiagonal matrix.

    reference -> http://dx.doi.org/10.1016/j.amc.2005.11.098

    py:
        python version prefered if True.

    *return*:
        Inv system instance.
    '''
    if tridmat.is_scalar:
        du=tridmat.__get_dl__(udl=True)
        dl=tridmat.__get_dl__(udl=False)
        u,v=tridmat.__get_uv__(du=du,dl=dl)
        return STInvSystem(u,v)
    n=tridmat.n
    p=tridmat.p
    placeholder=zeros([p,p])
    al=concatenate([[placeholder],-tridmat.lower],axis=0)
    bl=tridmat.diagonal
    cl=concatenate([[placeholder],-tridmat.upper],axis=0)
    if not py:
        ll1,ll2,ll3,ul1,ul2,ul3,hl1,hl2,hl3,invhl1,invhl2,invhl3=get_tlu_seq(al,bl,cl)
    else:
        #sequence for i < j
        ll1=al
        hl1=ones([n,p,p],dtype='complex128')*nan
        invhl1=ones([n,p,p],dtype='complex128')*nan
        ul1=ones([n,p,p],dtype='complex128')*nan
        for i in xrange(n-1):
            bi=bl[i]
            hi=bi
            if i!=0:
                hi=hi-ll1[i].dot(ui)
            invhi=inv(hi)
            hl1[i]=hi
            invhl1[i]=invhi
            ui=invhi.dot(cl[i+1])
            ul1[i]=ui
        #sequence for i > j
        ll2=cl
        hl2=ones([n,p,p],dtype='complex128')*nan
        invhl2=ones([n,p,p],dtype='complex128')*nan
        ul2=ones([n,p,p],dtype='complex128')*nan
        for i in xrange(n-1,-1,-1):
            hi=bl[i]
            if i!=n-1:
                hi=hi-ll2[i+1].dot(ui)
            invhi=inv(hi)
            hl2[i]=hi
            invhl2[i]=invhi
            if i!=0:
                ui=invhi.dot(al[i])
                ul2[i-1]=ui
        #sequence for i == j
        ll3=ll1
        ul3=[];hl3=[];invhl3=[]
        for i in xrange(n):
            hi=bl[i]
            if i!=0:
                hi=hi-ll3[i].dot(ul1[i-1])
            if i!=n-1:
                ui=invhl2[i+1].dot(al[i+1])
                ul3.append(ui)
                hi=hi-ll2[i+1].dot(ui)
            hl3.append(hi)
            invhl3.append(inv(hi))
    return BTInvSystem((ll1,hl1,ul1,invhl1),(ll2,hl2,ul2,invhl2),(ll3,hl3,ul3,invhl3))

def tinv(tridmat):
    '''
    get the inversion of this matrix.
    '''
    invs=get_inv_system(tridmat)
    return invs.get_all()


