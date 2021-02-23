'''Tests for the Question 2 of Coursework 1. RUN TO SEE IF PASS/FAIL'''
#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This year's modules\\CLA\\comp-lim-alg-course")
from cla_utils import *
import numpy as np



def test1(x,coefficients,f):
    """
    This test finds the difference between the true values of F(x) and the
    values of F(x) according to the approximated polynomial.
    If the difference between the true values of F(x) and approximated values
    of F(x) are zero (up to some negligible error), then the test passes.
    Else, it fails.
    """
    diff = PolyCoefficients(x, coefficients) - f
    error = np.inner(diff,diff)
    if error<1.0e-10:
        print("Test1 has passed")
    else:
        print("Test1 has failed")
        
#Running this file will produce the outcome of this test 

def LHS_Matrix(x,m):  
    """
    Creates LHS matrix as given in q2
    Given an nx1 numpy array compute a nxm matrix A where each row is a
    sequence of powers of an element of x up to the mth power, starting at 0
    :param x: nx1 numpy array
    :param m: integer
    :return A: nxm numpy array
    """
    n=len(x)
    
    A = np.zeros([n,m+1], dtype=x.dtype)
    
    for i in range(m+1):
       A[:,i] = np.power(x,i)
    
    return A 

def PolyCoefficients(x, coefficients):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x^0`` to ``x^m``).
    """
    m = len(coefficients)    
    y = 0
    for i in range(m):
        y += coefficients[i]*x**i
    return y

x = np.arange(-1,1.1,0.2)
f = 0.0*x
f[3:6] = 1
A_10 = householder_ls(LHS_Matrix(x,10), f)

#TEST IS RUN
test1(x,A_10,f)