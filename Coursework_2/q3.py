import numpy as np
#import os
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
from cla_utils import hessenberg
from cla_utils import pure_QR_mod
import matplotlib.pyplot as plt


def sign(x):
    
    if x == 0:
        sgn = 1
    else: 
        sgn = np.sign(x)
        
    return sgn


def qr_factor_tri(A):
    """
    Given a symmetric and tridiagonal matrix A, return V and A
    (working in place)

    :param A: an mxm-dimensional numpy array

    :return A: an mxm-dimensional numpy array containing the upper 
    triangular matrix
    :return V: 2x(m-1) matrix containing the 2- dimensional, normalised vk 
    needed to generate the Householder reflections
    """

    m, _ = A.shape
    
    #vector containing normalised vectors to generate Householder reflections
    V = np.zeros([2,m-1])
    e1 = np.zeros(2)
    e1[0] = 1
        
    for k in range(m-2):
        x = A[k:k+2,k] #only extract two components
        vk = sign(x[0])*np.sqrt(x.dot(x))*e1+x
        vk = vk/np.sqrt(vk.dot(vk))
        V[:,k] = vk
        F = np.eye(2)-2*np.outer(vk,vk.transpose()) #compute F matrix 
        A[k:k+2,k:k+2] = np.matmul(F,A[k:k+2,k:k+2])
        A[k:k+2,k+2] = F[:,1]*A[k+1,k+2]

    
    x = A[m-2:m,m-2] #only extract two components
    vk = sign(x[0])*np.sqrt(x.dot(x))*e1+x
    vk = vk/np.sqrt(vk.dot(vk))
    V[:,m-2] = vk
    F = np.eye(2)-2*np.outer(vk,vk.transpose())
    A[m-2:m,m-2:m] = np.matmul(F,A[m-2:m,m-2:m])
     

    return V,A


def qr_alg_tri(A):
    """
    Efficient implementation where reduced Q is computed (2x2) rather than
    whole Q (mxm) and matrix multiplication is carried out in place (2x2) or 
    (2x3)
    Given a symmetric and tridiagonal matrix T, return a similar matrix such
    that T[m,m-1] < 1e-12

    :param A: an mxm-dimensional numpy array

    :return T: similar matrix
    """
    
    
    m,_ = A.shape
    T = 1*A
    
    while np.abs(T[m-1,m-2])>1e-12:
        
        V,A = qr_factor_tri(A)
        
        # First two columns deal with 2x2 matrices
        vk = V[:,0]
        F = np.eye(2)-2*np.outer(vk,vk.transpose())
        Qreduced = F.T
        A[0:2,0:2] = np.matmul(A[0:2,0:2],Qreduced)
        
        #All columns thereafter deal with 3x2
        for i in range(1,m-1):
            vk = V[:,i]
            F = np.eye(2)-2*np.outer(vk,vk.transpose())
            Qreduced = F.T
            A[i-1:i+2,i:i+2] = np.matmul(A[i-1:i+2,i:i+2],Qreduced)
            
        T = 1*A    
     

    return T

def qr_alg_tri_mod(A):
    """
    Given a symmetric and tridiagonal matrix A, return a similar matrix such
    that T[m,m-1] < 1e-12

    :param A: an mxm-dimensional numpy array

    :return T: similar matrix
    :return Tn: sequence of errors at each iteration
    """
    
    Tn = []
    m,_ = A.shape
    T = 1*A
    Tn = np.append(Tn,np.abs(T[m-1,m-2]))
    
    while np.abs(T[m-1,m-2])>1e-12:
        V,A = qr_factor_tri(A)
        
        # First two columns deal with 2x2 matrices
        vk = V[:,0]
        F = np.eye(2)-2*np.outer(vk,vk.transpose())
        Qreduced = F.T
        A[0:2,0:2] = np.matmul(A[0:2,0:2],Qreduced)
        
        #All columns thereafter deal with 3x2
        for i in range(1,m-1):
            vk = V[:,i]
            F = np.eye(2)-2*np.outer(vk,vk.transpose())
            Qreduced = F.T
            A[i-1:i+2,i:i+2] = np.matmul(A[i-1:i+2,i:i+2],Qreduced)
        Tn = np.append(Tn,np.abs(A[m-1,m-2]))    
        T = 1*A   
        
    return T,Tn



#Create A_ij as in assignments using outer product (for efficiency)

X = np.ones(5)
Y = [x for x in range(1,6)]
A = np.outer(X,Y)+np.outer(Y,X)+np.ones([5,5])
A = np.reciprocal(A)
hessenberg(A) # this is tridiagonal as we expect 
A = qr_alg_tri(A)
D = np.diag(A) #these are an approximation of the eigenvalues

#plot eigenvalues
plt.figure(0)
plt.scatter(Y,D,c='black')
plt.title("Eigenvalues of A") 
plt.xlabel("ith biggest eigenvalue")
plt.ylabel("eigenvalue")
plt.xticks(Y)
plt.show()

def Eigval_iter(A):
    """
    Given a symmetric matrix, reduce it to Hessenberg form and then perform 
    the tridiag QR algorithm so as to extract the estimates of the eigenvalues 
    of the matrix. The absolute value of T(m,m-1) at each iteration is also 
    stored, concatenated at each submatrix and then returned 
    
    :param A: an mxm-dimensional numpy array

    :return eig: estimate of eigenvalues of A (mx1)
    :return iters: absolute values of T(m,m-1) up till tolerance is met at each
    tridiagonal matrix of row sizes m to 2 inclusive
    """
    m,_ = A.shape
    A = hessenberg(A)
    eig= np.zeros(m)
    iters = []
    
    for i in range(m-1,0,-1):
       
        A = A[:i+1,:i+1] #extract submatrix 
        A,Tn = qr_alg_tri_mod(A)
        eig[i] = A[i,i]
        iters = np.concatenate((iters,Tn),axis = None)
        
    eig[0] = A[0,0]

    return eig, iters

def iter_PureQR(A):
    """
    Given a symmetric matrix perform the pure QR algorithm 
    so as to extract absolute value of A(m,m-1) at each iteration is also 
    stored, concatenated at each submatrix and then returned 
    
    :param A: an mxm-dimensional numpy array

    :return iters: absolute values of T(m,m-1) up till tolerance is met at each
    matrix of row sizes m to 2 inclusive
    """
    m,_ = A.shape
    iters = []
    
    for i in range(m-1,0,-1):
       
        A = A[:i+1,:i+1] #extract submatrix 
        A,Tn = pure_QR_mod(A, maxit=10000, tol=1.0e-12)
        iters = np.concatenate((iters,Tn),axis = None)
        

    return iters

#plot concatenated errors
X = np.ones(5)
Y = [x for x in range(1,6)]
A = np.outer(X,Y)+np.outer(Y,X)+np.ones([5,5])
A = np.reciprocal(A)
A1 = 1*A
_,iters0 = Eigval_iter(A)
iters1 = iter_PureQR(A1)

plt.figure(1)
plt.plot(np.log10(iters0))
plt.title("Concatenated errors for tri-diag QR, m=5") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()

plt.figure(2)
plt.plot(np.log10(iters1))
plt.title("Concatenated errors for Pure QR, m=5") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()

X = np.ones(10)
Y = [x for x in range(1,11)]
A = np.outer(X,Y)+np.outer(Y,X)+np.ones([10,10])
A = np.reciprocal(A)
A1 = 1*A
_,iters0 = Eigval_iter(A)
iters1 = iter_PureQR(A1)

plt.figure(3)
plt.plot(np.log10(iters0))
plt.title("Concatenated errors for tri-diag QR, m=10") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()

plt.figure(4)
plt.plot(np.log10(iters1))
plt.title("Concatenated errors for Pure QR, m=10") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()

X = np.ones(15)
Y = [x for x in range(1,16)]
A = np.outer(X,Y)+np.outer(Y,X)+np.ones([15,15])
A = np.reciprocal(A)
A1 = 1*A
_,iters0 = Eigval_iter(A)
iters1 = iter_PureQR(A1)

plt.figure(3)
plt.plot(np.log10(iters0))
plt.title("Concatenated errors for tri-diag QR, m=15") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()

plt.figure(4)
plt.plot(np.log10(iters1))
plt.title("Concatenated errors for Pure QR, m=15") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()


def qr_alg_tri_wilk(A):
    """
    Given a symmetric and tridiagonal matrix A, return a similar matrix such
    that T[m,m-1] < 1e-12 using Wilkinson shift

    :param A: an mxm-dimensional numpy array

    :return T: similar matrix
    :return Tn: sequence of errors at each iteration
    """
    Tn = []
    m,_ = A.shape
    T = 1*A
    Tn = np.append(Tn,np.abs(T[m-1,m-2]))
    while np.abs(T[m-1,m-2])>1e-12:
        a = T[m-1,m-1]
        b = T[m-1,m-2]
        delta = (T[m-2,m-2]-T[m-1,m-1])/2
        mu = a - sign(delta)*(b**2)/(np.abs(delta)+np.sqrt(delta**2+b**2))
        V,A = qr_factor_tri(A-mu*np.eye(m))
        
        # First two columns deal with 2x2 matrices
        vk = V[:,0]
        F = np.eye(2)-2*np.outer(vk,vk.transpose())
        Qreduced = F.T
        A[0:2,0:2] = np.matmul(A[0:2,0:2],Qreduced)
        
        #All columns thereafter deal with 3x2
        for i in range(1,m-1):
            vk = V[:,i]
            F = np.eye(2)-2*np.outer(vk,vk.transpose())
            Qreduced = F.T
            A[i-1:i+2,i:i+2] = np.matmul(A[i-1:i+2,i:i+2],Qreduced)
        A  = A + mu*np.eye(m)
        Tn = np.append(Tn,np.abs(A[m-1,m-2]))    
        T = 1*A    
     

    return T,Tn



def Eigval_iter_wilk(A):
    """
    Given a symmetric matrix, reduce it to Hessenberg form and then perform 
    the tridiag QR algorithm so as to extract the estimates of the eigenvalues 
    of the matrix. The absolute value of T(m,m-1) at each iteration is also 
    stored, concatenated at each submatrix and then returned. Wilkinson shift 
    is used 
    
    :param A: an mxm-dimensional numpy array

    :return eig: estimate of eigenvalues of A (mx1)
    :return iters: absolute values of T(m,m-1) up till tolerance is met at each
    tridiagonal matrix of row sizes m to 2 inclusive
    """
    m,_ = A.shape
    A = hessenberg(A)
    eig= np.zeros(m)
    iters = []
    
    for i in range(m-1,0,-1):
       
        A = A[:i+1,:i+1] #extract submatrix 
        A,Tn = qr_alg_tri_wilk(A)
        eig[i] = A[i,i]
        iters = np.concatenate((iters,Tn),axis = None)
        
    eig[0] = A[0,0]

    return eig, iters

#plot concatenated errors when wilkinson shift is put in place
X = np.ones(5)
Y = [x for x in range(1,6)]
A = np.outer(X,Y)+np.outer(Y,X)+np.ones([5,5])
A = np.reciprocal(A)
_,iters0 = Eigval_iter_wilk(A)

plt.figure(5)
plt.plot(np.log10(iters0))
plt.title("Concatenated errors for tri-diag QR with Wilkinson shift, m=5") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()

X = np.ones(10)
Y = [x for x in range(1,11)]
A = np.outer(X,Y)+np.outer(Y,X)+np.ones([10,10])
A = np.reciprocal(A)
_,iters0 = Eigval_iter_wilk(A)

plt.figure(6)
plt.plot(np.log10(iters0))
plt.title("Concatenated errors for tri-diag QR with Wilkinson shift, m=10") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()

X = np.ones(15)
Y = [x for x in range(1,16)]
A = np.outer(X,Y)+np.outer(Y,X)+np.ones([15,15])
A = np.reciprocal(A)
_,iters0 = Eigval_iter_wilk(A)


plt.figure(7)
plt.plot(np.log10(iters0))
plt.title("Concatenated errors for tri-diag QR with Wilkinson shift, m=15") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()


#Construction of A = D+O and plots

D = [x for x in range(15,0,-1)]
A = np.diag(D) + np.ones([15,15])
A1 = 1*A
A2 = 1*A

eig0,iters0 = Eigval_iter(A)
iters1 = iter_PureQR(A1)
_,iters2 = Eigval_iter_wilk(A2)

plt.figure(8)
plt.plot(np.log10(iters0))
plt.title("Concatenated errors for tri-diag QR") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()

plt.figure(9)
plt.plot(np.log10(iters1))
plt.title("Concatenated errors for pure QR algorithm") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()

plt.figure(10)
plt.plot(np.log10(iters2))
plt.title("Concatenated errors for tri-diag QR with Wilkinson shift") 
plt.xlabel("index")
plt.ylabel("log base 10 of error")
plt.show()

#plot eigenvalues
plt.figure(11)
Y = [x for x in range(1,16)]
D = -np.sort(-eig0)
plt.scatter(Y,D,c='black')
plt.title("Eigenvalues of A") 
plt.xlabel("ith biggest eigenvalue")
plt.ylabel("eigenvalue")
plt.xticks(Y)
plt.show()