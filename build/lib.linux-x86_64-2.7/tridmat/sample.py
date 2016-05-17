import trid,linalg                  #import this library
from numpy.linalg import inv        #numpy inversion method
import time
n,p=100,10
tm=trid.get_trid(n,p,herm=True)     #construct a random block (np x np) hermitian tridiagonal matrix with block size p.
tma=tm.toarray()                    #parse this tridiagonal matrix to a numpy array.
t0=time.time()
isys=linalg.get_inv_system(tm)      #get the inversion generator.
inv00=isys[0,0]            #the the first element of the inverse matrix.
t1=time.time()
invtm=inv(tma)                      #the traditional method without optimization.
t2=time.time()
print 'Difference -> %s, Elapse %s(this), %s(numpy).'%(invtm[:p,:p]-inv00,t1-t0,t2-t1)
