#!/usr/bin/python
'''
Scalar and Block tridiagonalized matrices.
'''
from numpy import *
from scipy.sparse import csr_matrix,block_diag,coo_matrix,bsr_matrix
from scipy.sparse import bmat as sbmat
from scipy.sparse.linalg import inv as sinv
from numpy.linalg import inv
from futils.pywraper import ind2ptr,get_tlu_seq,ptr2ind
from utils import bcast_dot
from matplotlib.pyplot import *
import pdb,time

class TridMatrix(object):
    '''
    Tridiagonal matrix.

    diagonal:
        the diagonal part.
    upper:
        the upper diagonal part.
    '''
    def __init__(self,diagonal,upper,lower=None):
        self.diagonal=array(diagonal)
        self.upper=array(upper)
        assert(len(upper)==len(diagonal)-1)
        if lower is None:
            if self.is_scalar:
                self.lower=upper.conj()
            else:
                self.lower=swapaxes(upper,1,2).conj()
        else:
            self.lower=lower

    def __str__(self):
        n=self.n
        p=self.p
        sizestr='%s'%n if self.is_scalar else '%sx%s'%(n,p)
        return '''%s(%s):
upper -> %s
diagonal -> %s
lower -> %s
'''%(self.__class__,sizestr,self.upper,self.diagonal,self.lower)

    def __get_dl__(self,udl):
        '''
        get the diagonal part of UDL decomposition.

        udl:
            decompose as UDL if True else LDU
        '''
        n=self.n
        is_scalar=self.is_scalar
        al=self.diagonal #diagonal part
        ul=self.upper
        ll=self.lower
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

    def __get_uv__(self,du,dl):
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
        n=self.n
        #get vl
        bupper=-self.upper
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

    @property
    def p(self):
        '''block size.'''
        if self.is_scalar:
            return 1
        return self.diagonal.shape[-1]

    @property
    def n(self):
        '''number of blocks'''
        return self.diagonal.shape[0]

    @property
    def N(self):
        '''the dimension of this matrix.'''
        return self.p*self.n

    @property
    def shape(self):
        '''the shape of this matrix.'''
        N=self.N
        return (N,N)

    @property
    def is_scalar(self):
        '''return true if it is a scalar tridiagonal matrix.'''
        return ndim(self.diagonal)==1

    @property
    def dtype(self):
        '''the data type'''
        return self.upper.dtype

    def tocoo(self):
        '''transform to coo_matrix.'''
        raise Exception('Not Implemented!')

    def tocsr(self):
        '''transform to csr_matrix.'''
        raise Exception('Not Implemented!')

    def toarray(self):
        '''transform to an array.'''
        raise Exception('Not Implemented!')


class ScalarTridMatrix(TridMatrix):
    '''
    Scalar tridiagonal matrix class.
    '''
    def toarray(self):
        '''transform to array.'''
        n=self.n
        m=zeros((n,n),dtype=self.upper.dtype)
        fill_diagonal(m,self.diagonal)
        fill_diagonal(m[1:,:-1],self.lower)
        fill_diagonal(m[:-1,1:],self.upper)
        return m

    def tocoo(self):
        '''transform to coo_matrix.'''
        n=self.n
        indx=concatenate([arange(n-1),arange(n),arange(1,n)])
        indy=concatenate([arange(1,n),arange(n),arange(n-1)])
        data=concatenate([self.upper,self.diagonal,self.lower])
        return coo_matrix((data,(indx,indy)))

    def tocsr(self):
        '''transform to csr_matrix.'''
        return self.tocoo().tocsr()

    def toblocktrid(self):
        '''
        transform to block tridiagonal matrix.
        '''
        return BlockTridMatrix(self.diagonal[:,newaxis,newaxis],self.upper[:,newaxis,newaxis],self.lower[:,newaxis,newaxis])


class BlockTridMatrix(TridMatrix):
    '''
    Scalar tridiagonal matrix class.
    '''

    def toscalartrid(self):
        '''
        transform to block tridiagonal matrix.
        '''
        p=self.p
        if p!=1:
            raise Exception('Can not parse BlockTridMatrix to ScalarTridMatrix for block size p = %s!',p)
        return ScalarTridMatrix(self.diagonal[:,0,0],self.upper[:,0,0],self.lower[:,0,0])

    def tobsr(self):
        '''transform to bsr_matrix.'''
        n=self.n
        p=self.p
        m=ndarray((n,n),dtype='O')
        indx=concatenate([arange(n-1),arange(n),arange(1,n)])
        indy=concatenate([arange(1,n),arange(n),arange(n-1)])
        args=argsort(indx)
        data=concatenate([self.upper,self.diagonal,self.lower])
        res=bsr_matrix((data[args],indy[args],ind2ptr(indx[args],n)),blocksize=(p,p))
        return res

    def tocsr(self):
        '''transform to csr_matrix.'''
        return self.tobsr().tocsr()

    def tocoo(self):
        '''transform to coo_matrix.'''
        return self.tobsr().tocoo()

    def toarray(self):
        '''transform to an array.'''
        return self.tobsr().toarray()


def arr2trid(arr,p=None):
    '''
    Parse an array to a tridiagonal matrix.

    p:
        the block size, leave None to make it scalar.
    '''
    if p is None:
        return ScalarTridMatrix(arr.diagonal(),upper=arr.diagonal(1),lower=arr.diagonal(-1))
    else:
        N=len(arr)
        n=N/p
        dl=[]
        ul=[]
        ll=[]
        for i in xrange(n):
            dl.append(arr[i*p:(i+1)*p,i*p:(i+1)*p])
            if i!=n-1:
                ul.append(arr[i*p:(i+1)*p,(i+1)*p:(i+2)*p])
                ll.append(arr[(i+1)*p:(i+2)*p,i*p:(i+1)*p])
        return BlockTridMatrix(dl,upper=ul,lower=ll)

def sr2trid(mat,p=None):
    '''
    parse an bsr_matrix/csr_matrix instance to tridiagonal matrix.

    p:
        the block size, leave None to make it scalar.
    '''
    indptr=mat.indptr
    yindices=mat.indices
    is_block=hasattr(mat,'blocksize')
    if is_block:
        p=mat.blocksize[0]
        n=mat.shape[0]/p
        diagonal=zeros([n,p,p],dtype=mat.dtype)
        upper=zeros([n-1,p,p],dtype=mat.dtype)
        lower=zeros([n-1,p,p],dtype=mat.dtype)
    else:
        n=mat.shape[0]
        diagonal=zeros(n,dtype=mat.dtype)
        upper=zeros(n-1,dtype=mat.dtype)
        lower=zeros(n-1,dtype=mat.dtype)
    for i in xrange(n):
        x0=indptr[i]
        yinds=yindices[x0:x0+indptr[i+1]]
        for jind,j in enumerate(yinds):
            if j==i:
                diagonal[i]=mat.data[x0+jind]
            elif j==i-1:
                lower[j]=mat.data[x0+jind]
            elif j==i+1:
                upper[i]=mat.data[x0+jind]
    if is_block:
        return BlockTridMatrix(diagonal,upper=upper,lower=lower)
    else:
        return ScalarTridMatrix(diagonal,upper=upper,lower=lower)

def get_trid(n,p=None,fill=None,herm=False):
    '''
    Generate a random tridiagonal matrix.

    fill:
        the filling value, leave None for random numbers.
    p:
        the block size, leave None to make it scalar.
    herm:
        hermitian matrix if True.
    '''
    if fill is None:
        agen=random.random
    else:
        agen=lambda args: fill*ones(args)
    if p is None:
        dl=agen(n)
        ul=agen(n-1)+1j*agen(n-1)
        ll=lower=None if herm else (agen(n-1)+1j*agen(n-1))
        return ScalarTridMatrix(dl,upper=ul,lower=ll)

    ul=agen([n-1,p,p])+1j*agen([n-1,p,p])
    dl=agen([n,p,p])+1j*agen([n,p,p])
    if herm:
        ll=None
        dl=(dl+swapaxes(dl,1,2).conj())/2.
    else:
        ll=agen([n-1,p,p])+1j*agen([n-1,p,p])
    return BlockTridMatrix(dl,upper=ul,lower=ll)
