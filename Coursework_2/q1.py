import numpy as np


def tridiag(c, d):
    return np.diag(d, -1) + np.diag(c, 0) + np.diag(d, 1)


def LU_tridiag(c,d,b):
    """Solve the problem Ax = b with A given in a tridiagonal form and the LU 
    factorisation computed in place
    
    :param c,d: values on diagonals and sub/super diagonal of A
    :param b: RHS of Ax = b
    
    :return x: solution to Ax = b

    """
    
    m = len(b)
    
    Y = np.zeros(m)
    X = np.zeros(m)
    
    Y[0] = b[0]
    
    #store as 3 by m matrix for space efficiency
    
    T = np.zeros([3,m])
    
    #superdiagonal
    T[0,0:m-1] = d*np.ones(m-1)
    #subdiagonal
    T[2,0:m-1] = d*np.ones(m-1)
    #diagonal
    T[1,:] = c*np.ones(m)
    
    

    for k in range(m-1):
            
        T[2,k] /= T[1,k]
        
        #Merge forward substitution with LU factorisation
        
        Y[k+1] = b[k+1]- T[2,k]*Y[k]
            
        T[1,k+1] -= T[2,k]*T[0,k]
     
    #Backward substitution
     
    X[-1] = Y[-1]/T[1,m-1]
    
        
    for i in range(m-2,-1,-1):
       
        X[i] = (Y[i] - T[0,i]*X[i+1])/T[1,i]    
        
    return X   