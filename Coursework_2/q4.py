#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
import numpy as np
import numpy.random as random
from cla_utils import householder_ls
from cla_utils import operator_2_norm
from cla_utils import cond
from scipy.sparse import csgraph
import matplotlib.pyplot as plt

def back_subs(A, b):
    """
    For an upper triangular matrix A, solve Ax=b using backward substitution

    :param A: an mxm numpy array
    :param b: m dimensional numpy array

    :return x: solution to Ax = b
    """

    n = b.size

    x = np.zeros(n)

    for k in range(n-1,-1,-1):
        x[k] = (b[k] - np.dot(A[k,k+1:n],x[k+1:n]))/A[k,k]

    return x

#4a function
def GMRES_mod(A, b, maxit, tol, apply_pc = None, x0=None, return_residual_norms=False,
          return_residuals=False):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm with 
    preconditioning as an option 

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param apply_pc: None: no preconditioning (M=I), 0: M = diag(A), 1: M = triu(A)
     2: M = 1.1*triu(A)
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    """

    if x0 is None:
        x0 = b
        
    m, _ = A.shape
    Q = np.zeros([m, maxit+1])
    H = np.zeros((maxit+1, maxit))
    x = np.zeros(m)
    rnorms = np.zeros([maxit])

    
    nits = -1
    
    if apply_pc == 0:
        b = np.divide(b,np.diag(A))
    
    if apply_pc == 1:
        b = back_subs(np.triu(A),b)
        
    if apply_pc == 2:
        b = back_subs(np.triu(1.1*A),b)
    
    modb = np.linalg.norm(b)
    
    Q[:, 0] = b / modb
    
    for i in range(maxit):
        
        if apply_pc is None: 
           v = np.dot(A,Q[:,i])   
           
        if apply_pc == 0:
           v = np.divide(np.dot(A,Q[:,i]) ,np.diag(A))
    
        if apply_pc == 1:
           v = back_subs(np.triu(A),np.dot(A,Q[:,i]) )
           
        if apply_pc == 2:
           v = back_subs(np.triu(1.1*A),np.dot(A,Q[:,i]) )
             
        
        #vectorised inner for loop
        H[0:i+1,i] = np.matmul(Q[:,0:i+1].T,v)
        v -= np.matmul(Q[:,0:i+1],H[0:i+1,i])
        
        vnorm = np.linalg.norm(v)
        H[i+1,i] = vnorm
        Q[:, i+1] = v / vnorm
        
        e1 = np.zeros(i+2)
        e1[0] = 1
        #use householder from cla_utils to find solution which minimises
        y = householder_ls(H[0:i+2,0:i+1],modb*e1)
        
        x = np.dot(Q[:,0:i+1],y)
        
        rnorms[i] = np.linalg.norm(H[0:i+2, 0:i+1].dot(y) - modb*e1)
        
        if np.linalg.norm(H[0:i+2, 0:i+1].dot(y) - modb*e1) < tol:
            nits = i+1
            break
        
    if nits == -1:
        return x , nits , rnorms
    else:
        return x , nits , rnorms[0:nits]
         

def isDecreasing(A): 
    """
    For an array A, check to see if it is monotone decreasing.

    :param A: an mx1 numpy array


    :return: TRUE or FALSE
    """
    
    return (all(A[i] >= A[i + 1] for i in range(len(A) - 1))) 

#Create non trivial 4 by 4 matrix representing directed graph as referred to
#in write-up

G = np.arange(4) * np.arange(4)[:, np.newaxis]
G[0,1] = 5
G[1,0] = 5
G[1,1] = 0
G[2,2] = 0
G[3,3] = 0
L = csgraph.laplacian(G, normed=False)
A = L+np.eye(4)
M = 1.1*np.triu(A)
B = np.matmul(np.linalg.inv(M),A)
#check this is less than 1

norm = operator_2_norm(np.eye(4)-B) #value is 0.97

eigs0,V = np.linalg.eig(B)
eigs1,_ = np.linalg.eig(A)

cond0 = cond(V)
cond0 = np.sqrt(cond0)/100

x = max(np.abs(1-eigs0)) #0.81 (less than c=0.97, confirming proof)

random.seed(4)
b = random.randn(4) 


x0, _,r0 = GMRES_mod(A, b, maxit=100, tol=1.0e-10)

x1, _,r1 = GMRES_mod(A, b, maxit=100, tol=1.0e-10, apply_pc = 2)


#interim test to make sure both outputs are the same

if np.linalg.norm(x0-x1) < 1.0e-3:
    print('x0 and x1 are equivalent and have passed this interim test')


plt.figure(0)    
UpperBound = np.array([cond0*x,cond0*x**2,cond0*x**3,cond0*x**4])
plt.plot(np.array(range(1,5)), np.log10(r0),label='No Preconditioning')
plt.plot(np.array(range(1,5)), np.log10(r1),label='Preconditioning')
plt.plot(np.array(range(1,5)), np.log10(UpperBound),label='Log of Upper Bound for PC')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Log base 10 of residual")
plt.xticks(np.array(range(1,5)))
plt.title("Iteration vs logarithm of residual")
plt.show()

plt.figure(1)
plt.scatter(np.array(range(1,5)),np.real(eigs0),c='black', label = 'Preconditioned Eigenvalues')
plt.scatter(np.array(range(1,5)),np.real(eigs1),c='red', label = 'NO PC Eigenvalues')
plt.title("Eigenvalues of A and MinvA") 
plt.ylabel("eigenvalue")
plt.xticks(np.array(range(1,5)))
plt.legend()
plt.show()

