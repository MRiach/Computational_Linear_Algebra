'''Tests for question 5 of CW2.'''
#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-cw2-MRiach")
import pytest
import numpy as np
from q4 import isDecreasing
from q5 import analsol
from q5 import numsolq5
from numpy import random


#This tests to see if the numerical solution according to numsolq5 matches
#the analytical solution

@pytest.mark.parametrize('dt', [0.005, 0.01, 0.02])
def test_numsolq5(dt):

    

    u0 = analsol(0.01,dt)
    u1 = analsol(0.01,dt*2)
    u2 = analsol(0.01,dt*3)
    
    u3,_ = numsolq5(dt,0.01,1,0.1)
    


    assert(np.linalg.norm(u0-u3[:,0])< 1.0e-2)
    assert(np.linalg.norm(u1-u3[:,1])< 1.0e-2)
    assert(np.linalg.norm(u2-u3[:,2])< 1.0e-2)
        
 #This sees if the errors in the iterative scheme are decreasing   
@pytest.mark.parametrize('dt', [0.005, 0.01, 0.02])
def test_errordecreasing(dt):

    
    _,error = numsolq5(dt,0.01,1,0.1)
    


    assert(isDecreasing(error))


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)