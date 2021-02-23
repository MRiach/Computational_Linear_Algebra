'''Tests for question 4 of CW2.'''
#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-cw2-MRiach")
import pytest
import numpy as np
from q4 import GMRES_mod
from q4 import isDecreasing
from numpy import random


#This tests that the residuals are monotone decreasing and that if the tolerance
#is reached in the algorithm, then the output agrees with said tolerance

@pytest.mark.parametrize('m', [25, 30, 40])
def test_GMRES_mod(m):
    random.seed(4)
    A = random.randn(m, m)
    b = random.randn(m)

    x0, _,r0 = GMRES_mod(A, b, maxit=100, tol=1.0e-3)
    x1, _,r1 = GMRES_mod(A, b, maxit=100, tol=1.0e-3,apply_pc = 0)
    x2, _,r2 = GMRES_mod(A, b, maxit=100, tol=1.0e-3,apply_pc = 1)
    
    
    if len(r0) == 100:
        assert(isDecreasing(r0))
    else:    
        assert(np.linalg.norm(np.dot(A, x0) - b) < 1.0e-3)
        assert(isDecreasing(r0))
        
    if len(r1) == 100:
        assert(isDecreasing(r1))
    else:    
        assert(np.linalg.norm(np.dot(A, x1) - b) < 1.0e-3)
        assert(isDecreasing(r1))
        
    if len(r2) == 100:
        assert(isDecreasing(r2))
    else:    
        assert(np.linalg.norm(np.dot(A, x2) - b) < 1.0e-3)
        assert(isDecreasing(r2))
    


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)