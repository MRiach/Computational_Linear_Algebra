'''Tests for Mastery Component.'''
#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-mc-MRiach")
import pytest
import numpy as np
from polyfunctions import polyfitA
from polyfunctions import polyvalA
from polyfunctions import polyfit
from polyfunctions import polyval



#This tests to see that the Q used in polyfit is orthogonal and that solutions
#to both polyval and polyvalA agree up to a tolerance, as would be expected
#since polyvalA only marginally improves the fit.

@pytest.mark.parametrize('i', [5, 10, 15,20])
def test_mastery(i):
    
    X0 = 2*np.arange(10001)/10000-1
    X = np.cos(np.pi*np.arange(1001)/1000)
    Y = 1*X
    Y = 1+25*np.square(Y)
    Y = np.reciprocal(Y)
    
    
    c2 = polyfit(X,Y,i)
    Y1 = polyval(c2,X0)
    
    Q,H,d1 = polyfitA(X,Y,i)
    Y2 = polyvalA(d1,H,X0)
    
    #Take into account that Q is normed according to m^{1/2} rather than 1 
    #where m = 1001 in this case
    assert(np.linalg.norm((Q.T)@Q - 1001*np.eye(i+1)) < 1.0e-6)
    assert(np.linalg.norm(Y1-Y2) < 1.0e-6)
    

if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)