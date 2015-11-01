.. TridMat documentation master file, created by
   sphinx-quickstart on Fri Oct 30 22:13:03 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TridMat's documentation!
===================================

.. toctree::
   :maxdepth: 2

****************************************************
How to Build Scalar and Block Tridiagonal Matrices
****************************************************
Build with Functions
----------------
.. automodule:: trid
.. autofunction:: trid.arr2trid
.. autofunction:: trid.sr2trid
.. autofunction:: trid.get_trid
.. autofunction:: trid.build_sr

Data Types
----------------
.. autoclass:: trid.ScalarTridMatrix
   :members:
.. autoclass:: trid.BlockTridMatrix
   :members:

****************************************************
Linear Algebras(linalg/pylinalg)
****************************************************

*For Refences of Algorithms used in this section*,

* http://dx.doi.org/10.1137/0613045
* http://dx.doi.org/10.1016/j.amc.2005.11.098

.. automodule:: linalg
   :members:
LU Decomposition
----------------
.. autofunction:: linalg.trilu
.. autofunction:: linalg.trildu
.. autofunction:: linalg.triul
.. autofunction:: linalg.triudl

LU Data Types
----------------
.. automodule:: lusys
.. autoclass:: TLUSystem
   :members:

Inversion
----------------
.. autofunction:: linalg.tinv
.. autofunction:: linalg.get_inv_system
.. autofunction:: linalg.get_invh_system

Inv Data Types
----------------
.. automodule:: invsys
.. autoclass:: InvSystem
   :members:
.. autoclass:: BTInvSystem
   :members:
.. autoclass:: STInvSystem
   :members:

Indices and tables
==================
