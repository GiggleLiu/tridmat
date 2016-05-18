'''
Scalar and Block tridiagonalized matrices.

Also, there is some uitlities in this module that will help us build tridiagonal matrices.

A <TridMatrix> and it's derivatives can be parse to csr/coo matrices and array freely.
Some other matrix types are also supported.

In the following description, we take p -> the block dimension, N -> the matrix dimension and n = N/p.
'''

from numpy import *
from scipy.sparse import csr_matrix,block_diag,coo_matrix,bsr_matrix,issparse
from scipy.sparse import bmat as sbmat
from scipy.sparse.linalg import inv as sinv
from numpy.linalg import inv
import pdb

from futils.pywraper import ind2ptr,get_tlu_seq,ptr2ind

__all__=['TridMatrix','ScalarTridMatrix','BlockTridMatrix','arr2trid','sr2trid','get_trid','build_sr']

class TridMatrix(object):
    '''
    The base class for tridiagonal matrix with functionality partly realized.

    diagonal:
        Array of shape (n,p,p) -> block version or (n) -> scalar version,
        which is the diagonal part of this tridiagonal array.
    upper/lower:
        Array of shape (n-1,p,p) -> block version or (n-1) -> scalar version,
        which is the upper/lower part of this tridiagonal array.

        Leave lower `None` if your'd like it to be hermitian.
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

    @property
    def p(self):
        '''Block size.'''
        if self.is_scalar:
            return 1
        return self.diagonal.shape[-1]

    @property
    def n(self):
        '''Dimension of matrix in view of blocks'''
        return self.diagonal.shape[0]

    @property
    def N(self):
        '''The dimension of this matrix.'''
        return self.p*self.n

    @property
    def shape(self):
        '''The shape of this matrix.'''
        N=self.N
        return (N,N)

    @property
    def is_scalar(self):
        '''Return true if it is a scalar tridiagonal matrix.'''
        return ndim(self.diagonal)==1

    @property
    def dtype(self):
        '''Get the data type'''
        return self.upper.dtype

    def tocoo(self):
        '''Transform to coo_matrix.'''
        raise Exception('Not Implemented!')

    def tocsr(self):
        '''Transform to csr_matrix.'''
        raise Exception('Not Implemented!')

    def toarray(self):
        '''Transform to an array.'''
        raise Exception('Not Implemented!')


class ScalarTridMatrix(TridMatrix):
    '''
    Scalar tridiagonal matrix class.

    diagonal:
        Array of shape (n)
        which is the diagonal part of this tridiagonal array.
    upper/lower:
        Array of shape (n)
        which is the upper/lower part of this tridiagonal array.

        Leave lower `None` if your'd like it to be hermitian.
    '''
    def __init__(self,diagonal,upper,lower=None):
        assert(ndim(diagonal)==1)
        super(ScalarTridMatrix,self).__init__(diagonal,upper,lower)

    def toarray(self):
        '''Transform to array.'''
        n=self.n
        m=zeros((n,n),dtype=self.upper.dtype)
        fill_diagonal(m,self.diagonal)
        fill_diagonal(m[1:,:-1],self.lower)
        fill_diagonal(m[:-1,1:],self.upper)
        return m

    def tocoo(self):
        '''Transform to coo_matrix.'''
        n=self.n
        indx=concatenate([arange(n-1),arange(n),arange(1,n)])
        indy=concatenate([arange(1,n),arange(n),arange(n-1)])
        data=concatenate([self.upper,self.diagonal,self.lower])
        return coo_matrix((data,(indx,indy)))

    def tocsr(self):
        '''Transform to csr_matrix.'''
        return self.tocoo().tocsr()

    def toblocktrid(self):
        '''
        Transform to block tridiagonal matrix.
        '''
        return BlockTridMatrix(self.diagonal[:,newaxis,newaxis],self.upper[:,newaxis,newaxis],self.lower[:,newaxis,newaxis])


class BlockTridMatrix(TridMatrix):
    '''
    Scalar tridiagonal matrix class.

    diagonal:
        Array of shape (n,p,p)
        which is the diagonal part of this tridiagonal array.
    upper/lower:
        Array of shape (n,p,p)
        which is the upper/lower part of this tridiagonal array.

        Leave lower `None` if your'd like it to be hermitian.
    '''
    def __init__(self,diagonal,upper,lower=None):
        assert(ndim(diagonal)==3)
        super(BlockTridMatrix,self).__init__(diagonal,upper,lower)

    def toscalartrid(self):
        '''
        Transform to block tridiagonal matrix.
        '''
        p=self.p
        if p!=1:
            raise Exception('Can not parse BlockTridMatrix to ScalarTridMatrix for block size p = %s!',p)
        return ScalarTridMatrix(self.diagonal[:,0,0],self.upper[:,0,0],self.lower[:,0,0])

    def tobsr(self):
        '''Transform to bsr_matrix.'''
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
        '''Transform to csr_matrix.'''
        return self.tobsr().tocsr()

    def tocoo(self):
        '''Transform to coo_matrix.'''
        return self.tobsr().tocoo()

    def toarray(self):
        '''Transform to an array.'''
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
    Parse an bsr_matrix/csr_matrix instance to tridiagonal matrix.

    mat:
        An csr_matrix/bsr_matrix instance.
    p:
        The block size, leave None to make it scalar.

    *return*:
        A <ScalarTridMatrix> instance if p is None and mat is of type csr,
        otherwise, a <BlockTridMatrix> instance.
    '''
    if issparse(mat):
        if p is not None:
            if not isinstance(mat,bsr_matrix):
                mat=mat.tobsr((p,p))
            else:
                assert(mat.blocksize[0]==p)
        else:
            if isinstance(mat,bsr_matrix):
                p=mat.blocksize[0]
            elif not isinstance(mat,csr_matrix):
                mat=mat.tocsr()
    else:
        raise Exception('Sparse Matrix is required!')
    if p is not None:
        n=mat.shape[0]/p
        diagonal=zeros([n,p,p],dtype=mat.dtype)
        upper=zeros([n-1,p,p],dtype=mat.dtype)
        lower=zeros([n-1,p,p],dtype=mat.dtype)
    else:
        n=mat.shape[0]
        diagonal=zeros(n,dtype=mat.dtype)
        upper=zeros(n-1,dtype=mat.dtype)
        lower=zeros(n-1,dtype=mat.dtype)
    indptr=mat.indptr
    yindices=mat.indices
    for i in xrange(n):
        x0=indptr[i]
        yinds=yindices[x0:indptr[i+1]]
        for jind,j in enumerate(yinds):
            if j==i:
                #print mat[i,i]
                diagonal[i]=mat.data[x0+jind]
                #print diagonal[i]
            elif j==i-1:
                lower[j]=mat.data[x0+jind]
            elif j==i+1:
                upper[i]=mat.data[x0+jind]
    if p is not None:
        return BlockTridMatrix(diagonal,upper=upper,lower=lower)
    else:
        return ScalarTridMatrix(diagonal,upper=upper,lower=lower)

def get_trid(n,p=None,fill=None,herm=False):
    '''
    Generate a tridiagonal matrix.

    fill:
        The filling value
        
        Leave it None for random numbers.
    p:
        The block size.
        
        leave it None to make it scalar.
    herm:
        Get a hermitian matrix if True.

    *return*:
        A <ScalarTridMatrix> instance if p is None, else <BlockTridMatrix> instance
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

def build_sr(ll=None,dl=None,ul=None):
    '''
    Build a bsr or csr matrix by lower/diagonal/upper part of a tridiagonal matrix.

    ll/dl/ul:
        The lower/diagonal/upper part of tridiagonal matrix.

        Leave None to make any of them zeros(no all of them).

    *return*:
        a csr_matrix instance for scalar version, else a bsr_matrix instance.
    '''
    nzarr=None
    for i,il in enumerate([ll,ul,dl]):
        if il is not None:
            nzarr=il
            n=len(il)+1 if i!=2 else len(il)
            break
    if nzarr is None:
        raise ValueError('At least one of ll,dl,ul should be nonzeros!')
    is_scalar=ndim(nzarr)==1
    p=1 if is_scalar else nzarr.shape[-1]
    if is_scalar:
        mgen=csr_matrix
        nullval=zeros(0)
    else:
        mgen=bsr_matrix
        nullval=zeros([0,p,p])
    indx_d=arange(n)
    indx=concatenate([[] if dl is None else indx_d,[] if ll is None else indx_d[1:],[] if ul is None else indx_d[:-1]],axis=0)
    indy=concatenate([[] if dl is None else indx_d,[] if ll is None else indx_d[:-1],[] if ul is None else indx_d[1:]],axis=0)
    data=concatenate([nullval if dl is None else dl,nullval if ll is None else ll,nullval if ul is None else ul],axis=0)
    odl=argsort(indx)
    L=mgen((data[odl],indy[odl],ind2ptr(indx[odl],n)),dtype=nzarr.dtype)
    return L


