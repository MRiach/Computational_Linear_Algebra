#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-cw2-MRiach")
import numpy as np
from q1 import tridiag
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
import matplotlib.pyplot as plt

def numsolq5(dt,dx,T,alpha,stopcondition = 1e-5):
    """Output the numerical solution at all times starting at t=0 and
    ending at t=T for all time steps
    
    :param dt: time step
    :param dx: space step
    :param T: final time
    :alpha: parameter as per algorithm
    :stopcondition: error where iterative method terminates 
    
    :return U: numerical solution at all time steps up to T (MxN)
    :errors: sequence of errors at each iteration until stopping criterion is
             met 
    """
    M = 1/dx
    N = T/dt
    M,N = int(M), int(N)
    Errors = []
    
    #initial conditions
    X = np.array(range(1,M+1))/M
    p0 = np.cos(2*np.pi*X)+np.sin(2*np.pi*X)
    q0 = -2*np.pi*(np.cos(2*np.pi*X)+np.sin(2*np.pi*X))
    
    #set up of constants
    U0 = np.concatenate((p0,q0))
    U0 = U0.astype('complex')
    #initial guess is vector of zeros
    U = np.zeros(2*M*N,dtype = complex)
    B = np.zeros([2*M,2*M],dtype = complex)
    B[M:,0:M] = tridiag(-2*np.ones(M),1*np.ones(M-1))
    B[2*M-1,0] = 1
    B[M,M-1] = 1
    B[M:,0:M] = -dt/(dx**2)*B[M:,0:M] 
    B[0:M,M:] += -dt*np.eye(M)
    r = np.dot(np.eye(2*M)-0.5*B,U0)
    I = np.eye(2*M,dtype=complex)
    
    # set up of eigenvalues d_{1,k} and d_{2,k}
    d1 = np.exp(-2j*np.pi/N)**np.arange(N)
    d1 = d1*alpha**(1/N)
    d1 = 1-d1
    
    d2 = np.exp(-2j*np.pi/N)**np.arange(N)
    d2 = 1 + d2*alpha**(1/N)
    d2 = 0.5*d2
    
    #set up of diagonal which will be a N long array
    D = N*(alpha**(-1/N))**np.arange(N)
    D = D.astype('complex')
    error = 1


    while np.abs(error)>stopcondition:
        
        R = np.zeros(2*M*N,dtype = complex)
        R[0:2*M] = r - alpha*np.matmul((np.eye(2*M)-B/2),U[2*M*(N-1):])
        R = np.reshape(R,[N,2*M]).T
        #Multiplication by D^{-1}
        Rhat = np.multiply(R,np.reciprocal(D))
        Rhat = np.fft.fft(Rhat,axis=1)
        Uhat = np.zeros(2*M*N,dtype = complex)
        Uhat = np.reshape(Uhat,[N,2*M]).T
        
        for k in range(N):
            Uhat[:,k] = np.linalg.solve(d1[k]*I+d2[k]*B,Rhat[:,k]) 
            
        Uhat = np.fft.ifft(Uhat,axis=1)
        Uhat = np.multiply(Uhat,D)
        Unextiter = np.real(np.reshape(Uhat,2*M*N,order='F'))
        #evaluate difference between solution at each iteration and record this
        #as an error to see if stopcondition is met
        error = np.linalg.norm(U-Unextiter)
        Errors = np.append(Errors,error)
        U = 1*Unextiter
    
    
    U = np.reshape(U,[N,2*M]).T
    U = U[0:M,:]
    
    return U, Errors

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



#plot numerical and analytical solution for at T = 0.01, 0.02 and 0.03 
#with dx = 0.01

u0 = analsol(0.01,0.01)
u1 = analsol(0.01,0.02)
u2 = analsol(0.01,0.03)
u3,_ = numsolq5(0.01,0.01,1,0.1)
X = np.array(range(1,100+1))/100

plt.figure(0)   

plt.plot(X, u0)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Analytical solution to PDE at T=0.01")
plt.show()

plt.figure(1)   

plt.plot(X, u3[:,0])
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Numerical solution to PDE at T=0.01")
plt.show()

plt.figure(2)   

plt.plot(X, u1)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Analytical solution to PDE at T=0.02")
plt.show()

plt.figure(3)   

plt.plot(X, u3[:,1])
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Numerical solution to PDE at T=0.02")
plt.show()

plt.figure(4)   

plt.plot(X, u2)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Analytical solution to PDE at T=0.03")
plt.show()

plt.figure(5)   

plt.plot(X, u3[:,2])
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Numerical solution to PDE at T=0.03")
plt.show()
        
