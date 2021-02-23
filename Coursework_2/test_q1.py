'''Tests for question 1 of CW2.'''
#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-cw2-MRiach")
import pytest
import numpy as np
from q1 import tridiag
from q1 import LU_tridiag



#This tests to see that the solution to Ax = b is the same for the algorithm
#derived in q1 and and for the inverse algorithm used in python's linalg package

@pytest.mark.parametrize('m', [5, 10, 15])
def test_q1(m):
    
    b = np.ones(m)
    A = tridiag(1*np.ones(m),2*np.ones(m-1))
    Ainv = np.linalg.inv(A)
    
    x0 = LU_tridiag(1,2,b)
    x1 = np.dot(Ainv,b)
    assert(np.linalg.norm(x0-x1) < 1.0e-6)
    

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)