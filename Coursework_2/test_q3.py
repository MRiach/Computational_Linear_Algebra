'''Tests for question 3 of CW2.'''
#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-cw2-MRiach")
import pytest
import numpy as np
from q1 import tridiag
from q3 import qr_factor_tri
from q3 import qr_alg_tri
from q3 import Eigval_iter
from q3 import Eigval_iter_wilk
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
from cla_utils import householder

#Tests to see if output is the same as with householder in cla_utils and checks
#to see if upper triangular matrix that is outputted is upper triangular
@pytest.mark.parametrize('m', [5, 10, 15])
def test_qr_factor(m):
    
    A = tridiag(1*np.ones(m),2*np.ones(m-1))
    B = 1*A
    C = 1*A

    _,B = qr_factor_tri(B)
    C = householder(C)
    
    assert(np.allclose(B, np.triu(B)))
    assert(np.allclose(C, np.triu(C)))
    assert(np.allclose(B, C))
    
    
#Tests to see if the qr_alg_tri algorithm preserves trace, symmetry and rank    
@pytest.mark.parametrize('m', [10,15,20])
def test_qr_alg(m):
    
    A = tridiag(1*np.ones(m),2*np.ones(m-1))
    B = 1*A
    C = 1*A
    
    B = qr_alg_tri(B)
    
    assert(np.abs(np.trace(C) - np.trace(B)) < 1.0e-6)
    assert(np.allclose(B, B.T))
    assert(np.linalg.matrix_rank(C)==np.linalg.matrix_rank(B))

#Tests to see if the eigval_iter computes the eigenvalues of a symmetric matrix  
#successfully  
@pytest.mark.parametrize('m', [10,15,20])
def test_eigvaliter(m):
    
    X = np.ones(m)
    Y = [x for x in range(1,m+1)]
    A = np.outer(X,Y)+np.outer(Y,X)+np.ones([m,m])
    A = np.reciprocal(A)
    A1 = 1*A
    eig0,_ = np.linalg.eig(A)
    eig1,_ = Eigval_iter(A1)

    assert(np.allclose(eig0, eig1))

#Tests to see if the eigval_iter_wilk computes the eigenvalues of a symmetric matrix  
#successfully
@pytest.mark.parametrize('m', [10,15,20])
def test_eigvaliter_wilk(m):
    
    X = np.ones(m)
    Y = [x for x in range(1,m+1)]
    A = np.outer(X,Y)+np.outer(Y,X)+np.ones([m,m])
    A = np.reciprocal(A)
    A1 = 1*A
    eig0,_ = np.linalg.eig(A)
    eig1,_ = Eigval_iter_wilk(A1)

    assert(np.allclose(eig0, eig1))


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)