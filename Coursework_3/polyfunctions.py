#import os
import numpy as np
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
from cla_utils import householder_ls

def polyfit(X,f,n):
    """
    Find the vector c which minimises the residual of Ac = f using least 
    squares and the Vandermonde matrix directly
    
    :param X: Points used in interpolation
    :param f: Values of X at each function using interpolation
    :param n: degree of interpolating polynomial
    
    :return c: solution that minimises the residual
    """    
    A = np.vander(X,n+1,increasing = True)
    c = householder_ls(A,f)
    
    return c

def polyval(c,s):
    """
    Find the vector c which minimises the residual of Ac = f using least 
    squares and the Vandermonde matrix directly
    
    :param c: c as found in polyfit
    :param s: values to evaluate p(s)
    
    :return y: points evaluated at s
    """   
    n = len(c)
    B = np.vander(s,n,increasing = True)
    y = np.dot(B,c)

    return y

def polyfitA(X,f,n):
    """
    Find the vector d that minimises residual of Qd = f where 
    Q is a change of basis of X as per the Arnoldi algorithm.

    :param X: Points used in interpolation
    :param f: values of X at each function using interpolation
    :param: n: degree of interpolating polynomial
    
    
    :return Q: Matrix comprising q_i 
    :return H: Hessenberg matrix
    :return d: Least squares solution to Qd=f
    """

    m = len(X)
    H = np.zeros([n+1,n])
    Q = np.zeros([m,n+1])
    Q[:,0] = np.ones(m)
    
    for i in range(n):
        #piecewise component multiplication
        v = np.multiply(X,Q[:,i])
        H[0:i+1,i] = Q[:,0:i+1].T.dot(v/m)
        v -= Q[:,0:i+1].dot(H[0:i+1,i])
        norm = np.linalg.norm(v)
        #normalise using sqrt(m) as per Trefethen report
        H[i+1,i] = norm/np.sqrt(m)
        Q[:,i+1] = v/H[i+1,i]
    d = householder_ls(Q,f)
    
    return Q,H,d

def polyvalA(d,H,s):
    """
    Evaluate the polynomial approximated using the vector given by d at the 
    points in s, using the Hessenberg matrix, H, to form W.

    :param d: Least squares solution to Qd=f
    :param H: Hessenberg matrix which is similar to Vandermonde matrix created
              by elements of X
    :param: s: Values to evaluate p(s)
    
    
    :return y: Points evaluated at s
    """
    
    M = s.size
    n = H.shape[1]
    W = np.zeros([M,n+1])
    W[:,0] = np.ones(M)
    
    for i in range(n):
        w = np.multiply(W[:,i],s)
        w -= np.matmul(W[:,0:i+1], H[0:i+1,i])
        W[:, i+1] = w/H[i+1,i]
    
    y = np.matmul(W,d)
    return y 



