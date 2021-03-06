=======
# Elementwise inversion and LU decomposition for Block(Scalar) Tridiagonal Matrix

This library provide fast linear algebra implementations for Scalar and Block Tridiagonal Matrices.

They are:

* lu decomposition(O(np^3))
* ul decomposition(O(np^3))
* ldu decomposition(O(np^3))
* udl decomposition(O(np^3))
* fast inversion(O(xp^3))

Here, n is the matrix dimension with respect to blocks, and p the block size, N = n*p the true matrix idmension,
and x <= n is the number of matrix element you want to get.

The programming language is Python, with underlying fortran support.


###Purpose:
Tridiagonal Matrix, especially Block Tridiagonal Hermitian Matrices are frequently used data structure in physics.

It can be intepreted as a Hamiltonian of a chain Model and layered structures.

It's trivial implementation is the **recursive Green's function** method to get the surface Green's function, solve the **Landauer formula** to get the conductance.

This package provide an elementwise view of Green's function for a general tridiagonal matrix, in the language of matrix theory.

###To use
1. Install numpy/scipy
2. Download this repository, import and use it, e.g.
```python
import trid,linalg                  #import this library
from numpy.linalg import inv        #numpy inversion method
import time
n,p=100,10
tm=trid.get_trid(n,p,herm=True)     #construct a random block (np x np) hermitian tridiagonal matrix with block size p.
tma=tm.toarray()                    #parse this tridiagonal matrix to a numpy array.
t0=time.time()
isys=linalg.get_inv_system(tm)      #get the inversion generator.
inv00=isys.get_item(0,0)            #the the first element of the inverse matrix.
t1=time.time()
invtm=inv(tma)                      #the traditional method without optimization.
t2=time.time()
print 'Difference -> %s, Elapse %s(this), %s(numpy).'%(invtm[:p,:p]-inv00,t1-t0,t2-t1)
```
###To run tests/sample
        python tests.py
        python sample.py

###Contact the author: Leo
Email: cacate0129@gmail.com

Welcome to improve it!
