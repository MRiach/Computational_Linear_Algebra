#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-cw2-MRiach")
import numpy as np
from q1 import tridiag
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
from cla_utils import LU_inplace
from cla_utils import solve_L
from cla_utils import solve_U
import time as time
import matplotlib.pyplot as plt



def LU_inplace_mod(A):
    """Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.
    Also track the upper right and bottom left entries at each iteration

    :param A: an mxm-dimensional numpy array
    
    :return A: LU carried out inplace
    :topright: Array of values of top right entry at each iteration
    :bottomleft: Array of values of bottom left entry at each iteration

    """
    
    #This is the second and final attempt with slicing
    
    m = A.shape[0]
    topright = np.zeros(m)
    bottomleft = np.zeros(m)
    topright[0] = A[0,-1]
    bottomleft[0] = A[-1,0]

    for k in range(m-1):
            
            A[k+1:,k] /= A[k,k]
            
            A[k+1:,k+1:] -= np.outer(A[k+1:,k],A[k,k+1:])
            topright[k+1] = A[0,-1]
            bottomleft[k+1] = A[-1,0]
            
    return A, topright, bottomleft  

#matrices used in explanation in 2c
C = tridiag((1+2*2)*np.ones(10),-2*np.ones(9))
C[0,-1]=-2
C[-1,0]=-2
#keep original copy of C before LU inplace decomposition is carried out
C0 = 1*C

C,topright,bottomleft = LU_inplace_mod(C)
plt.figure(0)    
plt.plot(np.array(range(1,11)), topright,label='Top right entry')
plt.plot(np.array(range(1,11)), bottomleft,label='Bottom left entry')
plt.legend()
plt.xlabel("Iteration in LU decomposition algorithm")
plt.xlim([1,10])
plt.ylabel("Value of cell in respective entry")
plt.title("Values of specific entries in A at each iteration")
plt.show()

def LU_tridiag_mod(C,b):
    """Solve the problem Ax = b with A given in a tridiagonal form except for
    the top right and bottom left values and the LU factorisation 
    of the tridiagonal component of A is computed in place
    
    :param C: values used in matrix A (as per write up)
    :param b: RHS of Ax = b
    
    :return x: solution to Ax = b

    """
    
    M = len(b)
    
    Y = np.zeros(M)
    Z = np.zeros(M)
    X = np.zeros(M)
    
    Y[0] = b[0]
    
    #store as 3 by M matrix for space efficiency
    
    T = np.zeros([3,M])
    
    #superdiagonal
    T[0,0:M-1] = -C*np.ones(M-1)
    #subdiagonal
    T[2,0:M-1] = -C*np.ones(M-1)
    #diagonal
    T[1,:] = (1+2*C)*np.ones(M)
    

    for k in range(M-1):
            
        T[2,k] /= T[1,k]
        
        #Merge forward substitution with LU factorisation
        
        Y[k+1] = b[k+1]- T[2,k]*Y[k]
            
        T[1,k+1] -= T[2,k]*T[0,k]
        
    #Intermediate step of Z
  
    #Value in superdiagonal (which is constant and equivalent for every entry)
    #so we call this k
    k = T[0,0] 
    
    y = np.zeros(M-1)
    
    x = np.zeros(M)
    
    y[0] = -C/T[1,M-1]
    
    x[0] = -C/T[1,0]
    
    for i in range(1,M-1):
    
      y[i] = -T[2,i-1]*y[i-1]  #T[2,i-1]=L[i,i-1]

    
      x[i] = -k*x[i-1]/T[1,i]
    
    x[M-1] = -k*x[M-2]/T[1,M-1]-T[2,M-2]*y[M-2]  
    
    #construct inverse of (I_m+DE) as per write up 
    D = np.zeros([M,2])
    E = np.zeros([2,M])
    D[-1,1] = 1
    D[0:M-1,0] = y
    E[1,:] = x
    E[0,-1] = 1
    
    I = (np.eye(2)+np.matmul(E,D))
    
    #inverting 2 by 2 matrix is O(1) so very cheap computationally
    I = np.linalg.inv(I)

    I = np.matmul(D,I)

    I = np.matmul(I,E)
    
    I = np.eye(M)-I
    
    
    #This operation is approximately 2n^2 FLOPS
    
    Z = np.dot(I,Y)
    
    #Backward substitution
     
    X[-1] = Z[-1]/T[1,M-1]
    
        
    for i in range(M-2,-1,-1):
       
        X[i] = (Z[i] - T[0,i]*X[i+1])/T[1,i]    
        
    return X   

#Calculations to compare time

b = np.ones(1000)
b[0] = 5
A = tridiag((1+2*2)*np.ones(1000),-2*np.ones(1000-1))
A[0,-1]=-2
A[-1,0]=-2
A0=1*A
b0 = np.reshape(b,[1000,1])

start_time=time.time()
LU_tridiag_mod(2,b)
end_time=time.time()
times0 = end_time-start_time

start_time=time.time()
LU_inplace(A0)
y = solve_L(A0,b0,True)
x = solve_U(A0,y)
end_time=time.time()
times1 = end_time-start_time

times0=np.zeros(10)
times1=np.zeros(10)

for i in range(1,11):
    
    b = np.ones(i*100)
    b[0] = 5
    A = tridiag((1+2*2)*np.ones(i*100),-2*np.ones(i*100-1))
    A[0,-1]=-2
    A[-1,0]=-2
    A0=1*A
    b0 = np.reshape(b,[i*100,1])
    
    start_time=time.time()
    LU_tridiag_mod(2,b)
    end_time=time.time()
    times0[i-1]=end_time-start_time
    
    start_time=time.time()
    LU_inplace(A0)
    y = solve_L(A0,b0,True)
    x = solve_U(A0,y)
    end_time=time.time()
    times1[i-1] = end_time-start_time
    
plt.figure(1)    
plt.plot(100*np.array(range(1,11)), times0,label='LU tridiag')
plt.plot(100*np.array(range(1,11)), times1,label='LU general')
plt.legend()
plt.xlabel("M")
plt.xlim([100,1000])
plt.ylabel("Time")
plt.title("Size of square matrix vs Time taken to execute")
plt.show()

plt.figure(2)   
plt.plot(100*np.array(range(1,11)), np.log10(times0),label='LU tridiag')
plt.plot(100*np.array(range(1,11)), np.log10(times1),label='LU general')
plt.legend()
plt.xlabel("M")
plt.xlim([100,1000])
plt.ylabel("Log base 10 of Time")
plt.title("Size of square matrix vs Log Time taken to execute")
plt.show()

def numsol(dt,dx,T,Timesteps = np.zeros(0)):
    """Output the numerical solution at different times starting at t=0 and
    ending at t=T
    
    :param dt: time step
    :param dx: space step
    :param T: final time
    :Timesteps: Index of time steps that are to be plotted
    
    :return u: numerical solution at time t=T

    """
    C = dt**2/(4*dx**2)
    M = 1/dx
    N = T/dt
    M,N = int(M), int(N)
    
    #initial conditions
    X = np.array(range(1,M+1))/M
    u = np.cos(2*np.pi*X)+np.sin(2*np.pi*X)
    w0 = -2*np.pi*(np.cos(2*np.pi*X)+np.sin(2*np.pi*X))
    w1 = 1*w0
    B = tridiag(-2*np.ones(M),1*np.ones(M-1))
    B[0,-1] = 1
    B[-1,0] = 1
    
    #solutions are replaced and updated at each iteration to ensure efficiency
    for i in range(1,N+1):
        w0=1*w1
        b = w0+(dt/(dx**2))*np.dot(B,u)+(dt**2/(4*dx**2))*np.dot(B,w0)
        w1 = LU_tridiag_mod(C,b)
        u = u+dt/2*(w1+w0)
        if i in Timesteps:
            plt.figure(i)
            plt.plot(u)
        
    return u

def analsol(dx,T):
    """Output the analytical solution at t=T
    
    :param dx: space step
    :param T: time solution is obtained
    
    :return u: analytical solution at time t=T

    """
    M = 1/dx
    M= int(M)
    X = np.array(range(1,M+1))/M
    u = np.cos(2*np.pi*(X+T))+np.sin(2*np.pi*(X-T))
    
    return u

#plot numerical and analytical solution for T = 1

u0 = analsol(0.01,1)
u1 = numsol(0.01,0.01,1)
X = np.array(range(1,100+1))/100

plt.figure(3)   

plt.plot(X, u0)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Analytical solution to PDE at T=1")
plt.show()

plt.figure(4)   

plt.plot(X, u1)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Numerical solution to PDE at T=1")
plt.show()

#plot errors vs value of dx

errors=np.zeros(3)
X=np.array([0.01,0.05,0.1])

u0 = analsol(0.1,0.1)
u1 = numsol(0.01,0.1,0.1)
errors[2] = np.linalg.norm(u1-u0)

u0 = analsol(0.05,0.1)
u1 = numsol(0.01,0.05,0.1)
errors[1] = np.linalg.norm(u1-u0)

u0 = analsol(0.01,0.1)
u1 = numsol(0.01,0.01,0.1)
errors[0] = np.linalg.norm(u1-u0)

plt.figure(5)

plt.plot(X,errors)
plt.xlabel("dx")
plt.ylabel("Error")
plt.title("Error vs space step")
plt.show()