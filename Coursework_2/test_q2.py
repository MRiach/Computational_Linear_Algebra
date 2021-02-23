'''Tests for question 2 of CW2.'''
#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-cw2-MRiach")
import pytest
import numpy as np
from q1 import tridiag
from q2 import LU_tridiag_mod
from q2 import analsol
from q2 import numsol


#This tests to see that the solution to Ax = b is the same for the algorithm
#derived in q2f and and for the inverse algorithm used in python's linalg package

@pytest.mark.parametrize('m', [5, 10, 15])
def test_q2f(m):
    
    b = np.ones(m)
    A = tridiag((1+2*2)*np.ones(m),-2*np.ones(m-1))
    A[0,-1]=-2
    A[-1,0]=-2
    Ainv = np.linalg.inv(A)

    x0 = LU_tridiag_mod(2,b)
    x1 = np.dot(Ainv,b)
    assert(np.linalg.norm(x0-x1) < 1.0e-6)
    

#Tests to see numerical solution equals analytical solution 
@pytest.mark.parametrize('m', [0.5, 1, 1.5])
def test_q2g(m):
    
    u0 = analsol(0.01,m)
    u1 = numsol(0.01,0.01,m)

    assert(np.linalg.norm(u0-u1) < 1.0e-1)
    

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
    
    