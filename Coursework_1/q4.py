import os 
os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
#os.chdir("C:\\Users\\marwa\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
import numpy as np 
from cla_utils import *
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course\\clacourse-2020-cw1-MRiach")


rockdata = np.loadtxt('pressure.dat')

b = rockdata[:,1]
d = np.array([0,-5])
np.random.seed(0)
perturbation = np.random.normal(0,0.1,len(b))
b_perturbed = b + perturbation


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


#Here, we fix 10 points in the range(0,2) and compute the square of the 
#magnitude of the difference between P(x) and P1(x) at the ten points, where
#P1(x) is the polynomial that is formed when the pressure values are perturbed
#This allows us to investigate the sensitivity of the polynomials to their
#starting conditions, and will help us see which p is appropriate.

np.random.seed(0)
X = np.random.uniform(0,2,10)
Perturbdiff = np.zeros(49)
for i in range(1,50):
    PCoeff = LSSPolyCoeff(i,b = rockdata[:,1].reshape(100,1),d = np.array([0,-5]))
    P1Coeff = LSSPolyCoeff(i ,b = b_perturbed.reshape(100,1),d = np.array([0,-5]))
    P_values = np.zeros(10)
    P1_values = np.zeros(10)
    for j in range(10):
        if X[j]<1:
            P_values[j] =  PolyCoefficients(X[j],PCoeff[0:i+1])
            P1_values[j] =  PolyCoefficients(X[j],P1Coeff[0:i+1])
        else:
            P_values[j] =  PolyCoefficients(X[j],PCoeff[i+1:])
            P1_values[j] =  PolyCoefficients(X[j],P1Coeff[i+1:])
    Perturbdiff[i-1] = np.inner(P_values-P1_values,P_values-P1_values)
    

#This plot shows how the size of the change in values due to perturbation 
#varies with p. 
#Here, p=20 seems sensible as difference isn't stark and it appears to 
#sufficiently account for the data set 
plt.figure(0)
plt.plot(np.array(range(1,50)), Perturbdiff)
plt.title("Modulus Squared of Difference between Perturbed and Unperturbed Polynomials at 10 Points") 
plt.xlabel("p")
plt.ylabel("Difference")
plt.yscale('log')
plt.show()


#Plot of the perturbed AND unperturbed polynomials of degree 3 and 21 and 41
X1 = np.linspace(0,1,1000)
X2 = np.linspace(1,2,1000)
PCoeff = LSSPolyCoeff(3,b = rockdata[:,1].reshape(100,1),d = np.array([0,-5]))
P1Coeff = LSSPolyCoeff(3,b = b_perturbed.reshape(100,1),d = np.array([0,-5]))
P_1 = PolyCoefficients(X1,PCoeff[0:4])
P_2 = PolyCoefficients(X2,PCoeff[4:])
P1_1 = PolyCoefficients(X1,P1Coeff[0:4])
P1_2 = PolyCoefficients(X2,P1Coeff[4:])
plt.figure(1)
plt.xlim(0,2)
plt.plot(X1, P_1, color = 'blue', label = 'Unperturbed')
plt.plot(X2, P_2, color = 'blue')
plt.plot(X1, P1_1, color = 'red', label = 'Perturbed')
plt.plot(X2, P1_2, color = 'red')
plt.title("Polynomials of degree 3") 
plt.xlabel("depth, x")
plt.ylabel("Pressure, P(x)")
plt.legend()
plt.show()
 
X1 = np.linspace(0,1,1000)
X2 = np.linspace(1,2,1000)
PCoeff = LSSPolyCoeff(21,b = rockdata[:,1].reshape(100,1),d = np.array([0,-5]))
P1Coeff = LSSPolyCoeff(21,b = b_perturbed.reshape(100,1),d = np.array([0,-5]))
P_1 = PolyCoefficients(X1,PCoeff[0:22])
P_2 = PolyCoefficients(X2,PCoeff[22:])
P1_1 = PolyCoefficients(X1,P1Coeff[0:22])
P1_2 = PolyCoefficients(X2,P1Coeff[22:])
plt.figure(1)
plt.xlim(0,2)
plt.plot(X1, P_1, color = 'blue', label = 'Unperturbed')
plt.plot(X2, P_2, color = 'blue')
plt.plot(X1, P1_1, color = 'red', label = 'Perturbed')
plt.plot(X2, P1_2, color = 'red')
plt.title("Polynomials of degree 21") 
plt.xlabel("depth, x")
plt.ylabel("Pressure, P(x)")
plt.legend()
plt.show()

X1 = np.linspace(0,1,1000)
X2 = np.linspace(1,2,1000)
PCoeff = LSSPolyCoeff(41,b = rockdata[:,1].reshape(100,1),d = np.array([0,-5]))
P1Coeff = LSSPolyCoeff(41,b = b_perturbed.reshape(100,1),d = np.array([0,-5]))
P_1 = PolyCoefficients(X1,PCoeff[0:42])
P_2 = PolyCoefficients(X2,PCoeff[42:])
P1_1 = PolyCoefficients(X1,P1Coeff[0:42])
P1_2 = PolyCoefficients(X2,P1Coeff[42:])
plt.figure(2)
plt.xlim(0,2)
plt.plot(X1, P_1, color = 'blue', label = 'Unperturbed')
plt.plot(X2, P_2, color = 'blue')
plt.plot(X1, P1_1, color = 'red', label = 'Perturbed')
plt.plot(X2, P1_2, color = 'red')
plt.title("Polynomials of degree 41") 
plt.xlabel("depth, x")
plt.ylabel("Pressure, P(x)")
plt.legend()
plt.show()

#Plot the solutions
X1 = np.linspace(0,1,1000)
X2 = np.linspace(1,2,1000)
PCoeff = LSSPolyCoeff(3,b = rockdata[:,1].reshape(100,1),d = np.array([0,-5]))
P_1 = PolyCoefficients(X1,PCoeff[0:4])
P_2 = PolyCoefficients(X2,PCoeff[4:])
plt.figure(3)
plt.xlim(0,2)
plt.plot(X1, P_1, color = 'blue', label = 'Unperturbed')
plt.plot(X2, P_2, color = 'blue')
plt.scatter(rockdata[:,0],rockdata[:,1],c='black')
plt.title("Least Squares Solution, p = 3 with original data points") 
plt.xlabel("depth, x")
plt.ylabel("Pressure, P(x)")
plt.show()

X1 = np.linspace(0,1,1000)
X2 = np.linspace(1,2,1000)
PCoeff = LSSPolyCoeff(3,b = rockdata[:,1].reshape(100,1),d = np.array([0,-5]))
P_1 = PolyCoefficients(X1,PCoeff[0:4])
P_2 = PolyCoefficients(X2,PCoeff[4:])
plt.figure(4)
plt.xlim(0,2)
plt.plot(X1, P_1, color = 'blue', label = 'Unperturbed')
plt.plot(X2, P_2, color = 'blue')
plt.title("Least Squares Solution, p = 3") 
plt.xlabel("depth, x")
plt.ylabel("Pressure, P(x)")
plt.show()

X1 = np.linspace(0,1,1000)
X2 = np.linspace(1,2,1000)
PCoeff = LSSPolyCoeff(21,b = rockdata[:,1].reshape(100,1),d = np.array([0,-5]))
P_1 = PolyCoefficients(X1,PCoeff[0:22])
P_2 = PolyCoefficients(X2,PCoeff[22:])
plt.figure(3)
plt.xlim(0,2)
plt.plot(X1, P_1, color = 'blue', label = 'Unperturbed')
plt.plot(X2, P_2, color = 'blue')
plt.scatter(rockdata[:,0],rockdata[:,1],c='black')
plt.title("Least Squares Solution, p = 21 with original data points") 
plt.xlabel("depth, x")
plt.ylabel("Pressure, P(x)")
plt.show()

X1 = np.linspace(0,1,1000)
X2 = np.linspace(1,2,1000)
PCoeff = LSSPolyCoeff(21,b = rockdata[:,1].reshape(100,1),d = np.array([0,-5]))
P_1 = PolyCoefficients(X1,PCoeff[0:22])
P_2 = PolyCoefficients(X2,PCoeff[22:])
plt.figure(4)
plt.xlim(0,2)
plt.plot(X1, P_1, color = 'blue', label = 'Unperturbed')
plt.plot(X2, P_2, color = 'blue')
plt.title("Least Squares Solution, p = 21") 
plt.xlabel("depth, x")
plt.ylabel("Pressure, P(x)")
plt.show()

X1 = np.linspace(0,1,1000)
X2 = np.linspace(1,2,1000)
PCoeff = LSSPolyCoeff(41,b = rockdata[:,1].reshape(100,1),d = np.array([0,-5]))
P_1 = PolyCoefficients(X1,PCoeff[0:42])
P_2 = PolyCoefficients(X2,PCoeff[42:])
plt.figure(5)
plt.xlim(0,2)
plt.plot(X1, P_1, color = 'blue', label = 'Unperturbed')
plt.plot(X2, P_2, color = 'blue')
plt.scatter(rockdata[:,0],rockdata[:,1],c='black')
plt.title("Least Squares Solution, p = 41 with original data points") 
plt.xlabel("depth, x")
plt.ylabel("Pressure, P(x)")
plt.show()

X1 = np.linspace(0,1,1000)
X2 = np.linspace(1,2,1000)
PCoeff = LSSPolyCoeff(41,b = rockdata[:,1].reshape(100,1),d = np.array([0,-5]))
P_1 = PolyCoefficients(X1,PCoeff[0:42])
P_2 = PolyCoefficients(X2,PCoeff[42:])
plt.figure(6)
plt.xlim(0,2)
plt.plot(X1, P_1, color = 'blue', label = 'Unperturbed')
plt.plot(X2, P_2, color = 'blue')
plt.title("Least Squares Solution, p = 41") 
plt.xlabel("depth, x")
plt.ylabel("Pressure, P(x)")
plt.show()

        
#The set up of the system of equations in 4d is below
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

#Fill in values of B
B = np.zeros([8,12])

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

#Fill in values of d
d = np.zeros(8)


#carry out method to obtain x, as described in report
Q,R = householder_qr(B.T)

A1 = np.matmul(A,Q)[:,0:8]

A2 = np.matmul(A,Q)[:,9:]

y_1 = householder_solve(R.T[:,0:8], d)

y_2 = householder_ls(A2, b-np.dot(A1,y_1))

y = np.concatenate((y_1,y_2))

#obtain coefficients of polynomials of degree 2 
x = column_matvec(Q, y)


  

thetas = np.linspace(0,2*np.pi,201)

#Obtain the closed curve C 
C_a = np.sin(thetas)
C_b = 2*np.cos(thetas) + np.sin(thetas)

#Obtain values for C_hat so that they can be plotted alongside the curve
C_a_hat_0 = PolyCoefficients(thetas[0:101], x[[0,2,4]])
C_b_hat_0 = PolyCoefficients(thetas[0:101], x[[1,3,5]])

C_a_hat_1 = PolyCoefficients(thetas[100:], x[[6,8,10]])
C_b_hat_1 = PolyCoefficients(thetas[100:], x[[7,9,11]])


#Plots of C and C_hat
plt.figure(7)
plt.plot(C_a, C_b, color = 'blue', label = r'$C$')
plt.plot(C_a_hat_0, C_b_hat_0, color = 'red', label = r'$\hat{C}$')
plt.plot(C_a_hat_1, C_b_hat_1, color = 'red')
plt.title(r"Plot of C and $\hat{C}$") 
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#Plots of polynomials
plt.figure(8)
plt.ylim(-0.3,1)
plt.xlim(0,2*np.pi)
plt.plot(thetas[0:101], C_a_hat_0 , color = 'blue')
plt.plot(thetas[100:], C_a_hat_1 , color = 'blue')
plt.title(r"Plot of first component of $\hat{C}(\theta)$") 
plt.xlabel(r"$\theta$")
plt.ylabel(r"First component of $\hat{C}(\theta)$")
plt.show()

plt.figure(9)
plt.ylim(-0.028,-0.01)
plt.xlim(0,2*np.pi)
plt.plot(thetas[0:101], C_b_hat_0 , color = 'blue')
plt.plot(thetas[100:], C_b_hat_1 , color = 'blue')
plt.title(r"Plot of second component of $\hat{C}(\theta)$") 
plt.xlabel(r"$\theta$")
plt.ylabel(r"Second component of $\hat{C}(\theta)$")
plt.show()

