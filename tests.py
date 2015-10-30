#!/usr/bin/python
'''
Utilities for tridiagonal matrices.
'''
from numpy import *
from trid import ScalarTridMatrix,BlockTridMatrix,get_trid,arr2trid,sr2trid
import linalg as lin
import pdb,time
from scipy.sparse.linalg import inv as sinv
from scipy.sparse.linalg import splu
from numpy.linalg import inv
from matplotlib.pyplot import *

def test_lu(n,p):
    tm=get_trid(n,p)
    tmr=tm.tocoo().tocsr()
    tma=tmr.toarray()
    print 'checking for LU decomposition.'
    t0=time.time()
    lusys=lin.trilu(tm)
    L,U=lusys.L,lusys.U
    t1=time.time()
    L1,U=L.tocsr(),U.tocsc()
    lures=L1.dot(U)
    diff=abs(lures-tmr).toarray()
    print 'err -> %s, Elapse -> %s'%(diff.sum(),t1-t0)

    print 'checking for UL decomposition.'
    t0=time.time()
    ulsys=lin.triul(tm)
    U,L=lusys.L,lusys.U
    t1=time.time()
    U2,L=U.tocsr(),L.tocsc()
    ulres=U2.dot(L)
    diff=abs(ulres-tmr).toarray()
    print 'err -> %s, Elapse -> %s'%(diff.sum(),t1-t0)

    print 'checking for UDL decomposition.'
    t0=time.time()
    U,D,L=lin.udl(tm)
    t1=time.time()
    U,D3,L=U.tocsr(),D.tocsc(),L.tocsc()
    udl=U.dot(sinv(D3)).dot(L)
    print 'err -> %s, Elapse -> %s'%(abs(udl-tmr).sum(),t1-t0)

    print 'checking for LDU decomposition.'
    t0=time.time()
    L,D,U=lin.ldu(tm)
    t1=time.time()
    L,D4,U=L.tocsr(),D.tocsc(),U.tocsc()
    ldu=L.dot(sinv(D4)).dot(U)
    print 'err -> %s, Elapse -> %s'%(abs(ldu-tmr).sum(),t1-t0)
    pdb.set_trace()

def test_inv(n,p):
    tm=get_trid(n,p)
    tmr=tm.tocoo().tocsc()
    tma=tmr.toarray()
    print 'checking for inv'
    t0=time.time()
    res=lin.tinv(tm)
    t1=time.time()
    res2=sinv(tmr)
    t2=time.time()
    res3=inv(tma)
    t3=time.time()
    diff1=abs(res.dot(tma)-identity(tm.N)).sum()
    diff2=abs(res2.dot(tmr)-identity(tm.N)).sum()
    diff3=abs(res3.dot(tma)-identity(tm.N)).sum()
    print '(Trid, sps, dens) err -> %s(%s,%s), Elapse -> %s(%s,%s)'%(diff1,diff2,diff3,t1-t0,t2-t1,t3-t2)
    pdb.set_trace()

def test_dataparse():
    #test for sparse version
    a=random.random([10,10])
    smat=arr2trid(a)
    arrtrid=smat.toarray()
    spstrid=smat.tocsr()
    print arrtrid-spstrid
    #test for block version.
    smat=arr2trid(a,p=2)
    smat=sr2trid(smat.tobsr())
    print smat
    arrtrid=smat.toarray()
    spstrid=smat.tobsr()
    print arrtrid-spstrid


if __name__=='__main__':
    test_lu(500,2)
    #test_inv(200,None)
    #test_dataparse()
