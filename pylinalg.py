#!/usr/bin/python
'''
linalg lib for tridiagonal matrices, the python version.
'''
from numpy import *
from numpy.linalg import inv

def get_dl(tridmat,udl):
    '''
    get the diagonal part of UDL decomposition.

    udl:
        decompose as UDL if True else LDU
    '''
    n=tridmat.n
    is_scalar=tridmat.is_scalar
    al=tridmat.diagonal #diagonal part
    ul=tridmat.upper
    ll=tridmat.lower
    if is_scalar:
        cinv=lambda x:1./x
    else:
        cinv=inv
    if udl:
        al=al[::-1]
        ll=ll[::-1]
        ul=ul[::-1]
    dl=[]
    di=al[0]
    dl.append(di)
    for i in xrange(n-1):
        if udl:
            di=al[i+1]-dot(ul[i],dot(cinv(di),ll[i]))
        else:
            di=al[i+1]-dot(ll[i],dot(cinv(di),ul[i]))
        dl.append(di)
    if udl:
        dl.reverse()
    return array(dl)

def get_uv(tridmat,du,dl):
    '''
    get u,v vectors,
    the inversion of A is:
        inv(A) = [u1v1/2, u1v2 ... u1vn] + h.c.
                 [    , u2v2/2 ... u2vn]
                 [         ...         ]
                 [         ...   unvn/2]

    du/dl:
        the diagonal part of UDL and LDU decomposition.
    '''
    n=tridmat.n
    #get vl
    bupper=-tridmat.upper
    vi=1./du[0]
    vl=[vi]
    for i in xrange(1,n):
        vi=vi*bupper[i-1]/du[i]
        vl.append(vi)
    #get ul
    ui=1./dl[n-1]/vl[-1]
    ul=[ui]
    for i in xrange(n-2,-1,-1):
        ui=ui*bupper[i]/dl[i]
        ul.append(ui)
    ul.reverse()
    return array(ul),array(vl)

def get_inv_system(tridmat):
    '''
    get the sequence used in twist LU decomposition, A=LU.
    The pure python version.

    reference -> http://dx.doi.org/10.1016/j.amc.2005.11.098

    *return*:
        Inv system instance.
    '''
    n=tridmat.n
    p=tridmat.p
    placeholder=ones([p,p])
    bl=tridmat.diagonal
    al=concatenate([[placeholder],-tridmat.lower],axis=0)
    cl=-tridmat.upper
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
        ui=invhi.dot(cl[i])
        ul1[i]=ui
    #sequence for i > j
    ll2=concatenate([[placeholder],cl],axis=0)
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
    n=tridmat.n
    p=tridmat.p
    al=concatenate([tridmat.lower],axis=0)
    bl=tridmat.diagonal
    cl=concatenate([tridmat.upper],axis=0)
    if tridmat.is_scalar:
        cinv=lambda x:1./x
    else:
        cinv=inv
    #sequence for i < j
    ll1=al
    hl1=ones([n,p,p],dtype='complex128')*nan
    invhl1=ones([n,p,p],dtype='complex128')*nan
    ul1=ones([n,p,p],dtype='complex128')*nan
    for i in xrange(n-1):
        bi=bl[i]
        hi=bi
        if i!=0:
            hi=hi-dot(ll1[i-1],ui)
        invhi=cinv(hi)
        hl1[i]=hi
        invhl1[i]=invhi
        ui=invhi.dot(cl[i])
        ul1[i]=ui
    #sequence for i > j
    ll2=cl
    hl2=ones([n,p,p],dtype='complex128')*nan
    invhl2=ones([n,p,p],dtype='complex128')*nan
    ul2=ones([n,p,p],dtype='complex128')*nan
    for i in xrange(n-1,-1,-1):
        hi=bl[i]
        if i!=n-1:
            hi=hi-ll2[i].dot(ui)
        invhi=cinv(hi)
        hl2[i]=hi
        invhl2[i]=invhi
        if i!=0:
            ui=invhi.dot(al[i-1])
            ul2[i-1]=ui
    #sequence for i == j
    ll3=ll1
    ul3=[];hl3=[];invhl3=[]
    for i in xrange(n):
        hi=bl[i]
        if i!=0:
            hi=hi-ll3[i-1].dot(ul1[i-1])
        if i!=n-1:
            ui=invhl2[i+1].dot(al[i])
            ul3.append(ui)
            hi=hi-ll2[i].dot(ui)
        hl3.append(hi)
        invhl3.append(cinv(hi))
    return BTInvSystem((ll1,hl1,ul1,invhl1),(ll2,hl2,ul2,invhl2),(ll3,hl3,ul3,invhl3))

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
 
