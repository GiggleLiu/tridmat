'''
Tests for tridiagonal matrices.
'''

from numpy import *
import pdb,time
from scipy.sparse.linalg import inv as sinv
from scipy.sparse.linalg import splu
from numpy.linalg import inv
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose

from trid import ScalarTridMatrix,BlockTridMatrix,get_trid,arr2trid,sr2trid
import linalg as lin

class Tester(object):
    def __init__(self):
        n,p=500,2
        self.tm=get_trid(n,p)
        self.tmr=self.tm.tocoo().tocsr()
        self.tma=self.tmr.toarray()

    def test_lu(self):
        tm,tmr,tma=self.tm,self.tmr,self.tma
        print 'checking for LU decomposition.'
        t0=time.time()
        lusys=lin.trilu(tm)
        L,U=lusys.L,lusys.U
        t1=time.time()
        L1,U=L.tocsr(),U.tocsc()
        lures=L1.dot(U)
        diff=abs(lures-tmr).toarray()
        print 'err -> %s, Elapse -> %s'%(diff.sum(),t1-t0)
        assert_allclose(diff,0,atol=1e-10)

    def test_ul(self):
        print 'checking for UL decomposition.'
        tm,tmr,tma=self.tm,self.tmr,self.tma
        lusys=lin.trilu(tm)
        t0=time.time()
        ulsys=lin.triul(tm)
        U,L=lusys.L,lusys.U
        t1=time.time()
        U2,L=U.tocsr(),L.tocsc()
        ulres=U2.dot(L)
        diff=abs(ulres-tmr).toarray()
        print 'err -> %s, Elapse -> %s'%(diff.sum(),t1-t0)
        assert_allclose(diff,0,atol=1e-10)

    def test_udl(self):
        print 'checking for UDL decomposition.'
        tm,tmr,tma=self.tm,self.tmr,self.tma
        t0=time.time()
        udlsys=lin.triudl(tm)
        U,D,L=udlsys.U,udlsys.D,udlsys.L
        t1=time.time()
        U,D3,L=U.tocsr(),D.tocsc(),L.tocsc()
        udl=U.dot(D3).dot(L)
        print 'err -> %s, Elapse -> %s'%(abs(udl-tmr).sum(),t1-t0)
        assert_allclose(udl.toarray(),tma,atol=1e-10)

    def test_ldu(self):
        print 'checking for LDU decomposition.'
        tm,tmr,tma=self.tm,self.tmr,self.tma
        t0=time.time()
        ldusys=lin.trildu(tm)
        L,D,U=ldusys.L,ldusys.D,ldusys.U
        t1=time.time()
        L,D4,U=L.tocsr(),D.tocsc(),U.tocsc()
        ldu=L.dot(D4).dot(U)
        print 'err -> %s, Elapse -> %s'%(abs(ldu-tmr).sum(),t1-t0)
        assert_allclose(ldu.toarray(),tma,atol=1e-10)

    def test_inv(self):
        n,p=200,2
        tm=get_trid(n,p,herm=True)
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
        assert_almost_equal(diff1,0)
        assert_almost_equal(diff2,0)
        assert_almost_equal(diff3,0)

    def test_tLU(self):
        n,p=500,2
        tm=get_trid(n,p,herm=True)
        tmr=tm.tocoo().tocsc()
        tma=tmr.toarray()
        print 'checking for twist LU decomposition'
        t0=time.time()
        invegn=lin.get_inv_system(tm)
        L,U=invegn.get_twist_LU(j=2)
        t1=time.time()
        diff=abs(L.dot(U)-tmr).sum()
        print 'err -> %s, Elapse -> %s'%(diff,t1-t0)
        assert_almost_equal(diff,0)

    def test_all(self):
        self.test_lu()
        self.test_ul()
        self.test_udl()
        self.test_ldu()
        self.test_inv()
        self.test_tLU()

class TestParse(object):
    def __init__(self):
        n,p=500,3
        self.n=n
        self.p=p
        self.btrid=BlockTridMatrix(diagonal=random.random([n,p,p]),\
                upper=random.random([n-1,p,p]),lower=random.random([n-1,p,p]))
        self.strid=ScalarTridMatrix(diagonal=random.random(n),upper=random.random(n-1),lower=random.random(n-1))

    def test_a2trid(self):
        #for scalar
        print 'Testing for conversion between ScalarTridMatrix and array.'
        a=self.strid.toarray()
        smat=arr2trid(a)
        a2=smat.toarray()
        assert_allclose(a,a2,atol=1e-10)
        #for block
        print 'Testing for conversion between BlockTridMatrix and array.'
        a=self.btrid.toarray()
        smat=arr2trid(a,p=self.p)
        a2=smat.toarray()
        assert_allclose(a,a2,atol=1e-10)

    def test_csr2trid(self):
        #for scalar
        print 'Testing for conversion between ScalarTridMatrix and CSR.'
        a=self.strid.tocsr()
        smat=sr2trid(a)
        smat2=arr2trid(self.strid.toarray())
        a2=smat.toarray()
        assert_allclose(a.toarray(),a2,atol=1e-10)
        #for block
        print 'Testing for conversion between BlockTridMatrix and CSR.'
        a=self.btrid.tocsr()
        smat=sr2trid(a,p=self.p)
        a2=smat.toarray()
        assert_allclose(a.toarray(),a2,atol=1e-10)

    def test_bsr2trid(self):
        print 'Testing for conversion between BlockTridMatrix and BSR.'
        a=self.btrid.tobsr()
        smat=sr2trid(a,p=self.p)
        a2=smat.toarray()
        assert_allclose(a.toarray(),a2,atol=1e-10)

    def test_all(self):
        self.test_a2trid()
        self.test_csr2trid()
        self.test_bsr2trid()

if __name__=='__main__':
    TestParse().test_all()
    Tester().test_all()

