'''Tests for the Question 4 of Coursework 1. RUN TO SEE IF PASS/FAIL'''
import os 
os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
from cla_utils import *
import numpy as np
os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course\\clacourse-2020-cw1-MRiach")
rockdata = np.loadtxt('pressure.dat')

pi=np.pi
#Fill in A row by row
A = np.zeros([16,12])

A[0,0] = 1

A[1,1] = 1

A[2,0] = 1
A[2,2] = pi/4
A[2,4] = (pi/4)**2

A[3,1] = 1
A[3,3] = pi/4
A[3,5] = (pi/4)**2

A[4,0] = 1
A[4,2] = pi/2
A[4,4] = (pi/2)**2

A[5,1] = 1
A[5,3] = pi/2
A[5,5] = (pi/2)**2

A[6,0] = 1
A[6,2] = 3*pi/4
A[6,4] = (3*pi/4)**2

A[7,1] = 1
A[7,3] = 3*pi/4
A[7,5] = (3*pi/4)**2 

A[8,6] = 1
A[8,8] = pi
A[8,10] = pi**2

A[9,7] = 1
A[9,9] = pi
A[9,11] = pi**2

A[10,6] = 1
A[10,8] = 3*pi/2
A[10,10] = (3*pi/2)**2

A[11,7] = 1
A[11,9] = 3*pi/2
A[11,11] = (3*pi/2)**2

A[12,6] = 1
A[12,8] = 5*pi/3
A[12,10] = (5*pi/3)**2

A[13,7] = 1
A[13,9] = 5*pi/3
A[13,11] = (5*pi/3)**2

A[14,6] = 1
A[14,8] = 7*pi/4
A[14,10] = (7*pi/4)**2

A[15,7] = 1
A[15,9] = 7*pi/4
A[15,11] = (7*pi/4)**2

#Fill in values of b
b = np.array([0, 2, np.sqrt(2)/2, 3*np.sqrt(2)/2, 1, 1, np.sqrt(2)/2,
              -np.sqrt(2)/2, 0, -2, -1, -1, -np.sqrt(3)/2,
              -np.sqrt(3)/2+1, -np.sqrt(2)/2, np.sqrt(2)/2 ])
b = b.reshape(16,1)


B = np.zeros([8,12])
pi = np.pi
B[0,0] = 1
B[1,1] = 1
B[0,6] = -1
B[1,7] = -1
B[0,8] = -2*pi
B[1,9] = -2*pi
B[0,10] = -(2*pi)**2
B[1,11] = -(2*pi)**2

B[2,2] = 1
B[3,3] = 1
B[2,8] = -1
B[3,9] = -1
B[2,10] = -4*pi
B[3,11] = -4*pi

B[4,0] = 1
B[5,1] = 1
B[4,2] = pi
B[5,3] = pi
B[4,4] = pi**2
B[5,5] = pi**2
B[4,6] = -1
B[5,7] = -1
B[4,8] = -pi
B[5,9] = -pi
B[4,10] = -pi**2
B[5,11] = -pi**2

B[6,2] = 1
B[7,3] = 1
B[6,4] = 2*pi
B[7,5] = 2*pi
B[6,8] = -1
B[7,9] = -1
B[6,10] = -2*pi
B[7,11] = -2*pi


d = np.zeros(8)

Q,R = householder_qr(B.T)

A1 = np.matmul(A,Q)[:,0:8]

A2 = np.matmul(A,Q)[:,9:]

y_1 = householder_solve(R.T[:,0:8], d)

y_2 = householder_ls(A2, b-np.dot(A1,y_1))

y = np.concatenate((y_1,y_2))

#obtain coefficients of polynomials of degree 2 
x = column_matvec(Q, y)

def get_A_Matrix(p):
    """
    Creates A matrix as given in q4a
    """       
    A = np.zeros([100,2*(p+1)], dtype=float)
    
    for i in range(p+1):
       A[0:50,i] = np.power(rockdata[0:50,0],i)
       A[50:100,i+p+1] = np.power(rockdata[50:100,0],i)
    
    return A 

def get_B_Matrix(p):
    """
    Creates B matrix as given in q4a 
    """
         
    B = np.zeros([2,2*(p+1)], dtype=float)
    
    B[0,0:p+1] = np.ones(p+1)
    B[0,p+1:2*(p+1)] = -1*np.ones(p+1)
    B[1,0:p+1] = np.array(range(p+1))
    B[1,p+1:2*(p+1)] = -1*np.array(range(p+1))
    
    return B 

def LSSPolyCoeff(p,b = rockdata[:,1].reshape(100,1),d = np.array([0,-5])):
    """
    #Returns solution x (the coefficients of the polynomials) to the least squares
    problem given in 4a where the polynomials are of degree p        
    """
    A = get_A_Matrix(p)

    B = get_B_Matrix(p)

    Q,R = householder_qr(B.T)

    A1 = np.matmul(A,Q)[:,0:2]

    A2 = np.matmul(A,Q)[:,2:]

    y_1 = householder_solve(R.T[0:2,0:2], d)

    y_2 = householder_ls(A2, b-np.dot(A1,y_1))

    y = np.concatenate((y_1,y_2))

    x = column_matvec(Q, y)

    
    return x 

def PolyCoefficients(x, coefficients):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x^0`` to ``x^m``).
    """
    m = len(coefficients)    
    y = 0
    for i in range(m):
        y += coefficients[i]*x**i
    return y

def test2(p=10,d = np.array([0,-5])):
    """
    This test verifies if the constraints are met by the LS solution for p=10.
    """
    B = get_B_Matrix(p)
    
    x = LSSPolyCoeff(p)
    
    #Bx - d = 0 => constraints are met 
    
    diff = column_matvec(B, x) - d
    
    error = np.inner(diff,diff)
    
    if error<1.0e-10:
        print("Test2 has passed")
    else:
        print("Test2 has failed")
        
def test3(p=1,b = rockdata[:,1]):
    """
    This test makes sure other vectors,y, that satisfy the constraints yield
    a larger value for ||Ay-b||^2 than the solution, x, we obtain with 
    LSSPolyCoeff for p=1
    """
    A = get_A_Matrix(p)
    
    x = LSSPolyCoeff(p)
    
    #y1 and y2 both satisfy the constraints of the problem
    
    y1 = np.array([5,0,0,5])
    
    y2 = np.array([2.7,-2,-2.3,3])
        
    diff0 = column_matvec(A, x) - b
    
    diff1 = column_matvec(A, y1) - b
    
    diff2 = column_matvec(A, y2) - b
    
    error0 = np.inner(diff0,diff0)
    
    error1 = np.inner(diff1,diff1)
    
    error2 = np.inner(diff2,diff2)  
    
    if error0<error1 and error0<error2:
        print("Test3 has passed")
    else:
        print("Test3 has failed")



def test4(x=x,d = d, B=B):
    """
    This test verifies if the constraints are met by the LS solution in 4d.
    """   
    #Bx - d = 0 => constraints are met 
    
    diff = column_matvec(B, x) - d
    
    error = np.inner(diff,diff)
    
    if error<1.0e-10:
        print("Test4 has passed")
    else:
        print("Test4 has failed")

#TESTS ARE RUN 
test2()
test3()
test4()