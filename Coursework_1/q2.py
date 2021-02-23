#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This year's modules\\CLA\\comp-lim-alg-course")
#os.chdir("C:\\Users\\marwa\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This year's modules\\CLA\\comp-lim-alg-course")
import numpy as np 
from cla_utils import *
import matplotlib.pyplot as plt

x = np.arange(-1,1.1,0.2)
f = 0.0*x
f[3:6] = 1


def LHS_Matrix(x,m):  
    """
    Creates LHS matrix as given in q2
    Given an nx1 numpy array compute a nxm matrix A where each row is a
    sequence of powers of an element of x up to the mth power, starting at 0
    :param x: nx1 numpy array
    :param m: integer
    :return A: nxm numpy array
    """
    n=len(x)
    
    A = np.zeros([n,m+1], dtype=x.dtype)
    
    for i in range(m+1):
       A[:,i] = np.power(x,i)
    
    return A 

def PolyCoefficients(x, coefficients):
    """ Returns a polynomial for ``x`` values for the ``coeffs`` provided.

    The coefficients must be in ascending order (``x^0`` to ``x^m``).
    """
    m = len(coefficients)    
    y = 0
    for i in range(m):
        y += coefficients[i]*x**i
    return y

#Use householder algorithm from exercises 3 to compute the least squares
#solution for the a_i coefficients

A_10 = householder_ls(LHS_Matrix(x,10), f)
print("The m=10 LSS a_i coefficients are: " + str(A_10))

#Plot the polynomial just outside the range of the x values to see its 
#behaviour elsewhere 
X = np.linspace(-1.04,1.04,1000)
plt.figure(0)
plt.plot(X, PolyCoefficients(X, A_10))
plt.scatter(x,f,c='black')
plt.title("Approximation of F using the QR method and a polynomial of degree 10") 
plt.xlabel("x")
plt.ylabel("F(x)")
plt.show()

#Perturbed f with fixed seed so this can be replicated 
np.random.seed(0)
perturbation = np.random.normal(0,0.1,len(f))
f_perturbed = f + perturbation

#Use householder algorithm from exercises 3 to compute the least squares
#solution for the a_i coefficients
A_10_perturbed = householder_ls(LHS_Matrix(x,10), f_perturbed)
print("The m= 10 LSS perturbed a_i coefficients are: " + str(A_10_perturbed))

#Plot the polynomial just outside the range of the x values to see its 
#behaviour elsewhere 
X = np.linspace(-1.04,1.04,1000)
plt.figure(1)
plt.plot(X, PolyCoefficients(X, A_10_perturbed), label = 'Perturbed F')
plt.plot(X, PolyCoefficients(X, A_10), label = 'F')
plt.title("Approximation of perturbed F using the QR method and a polynomial of degree 10") 
plt.xlabel("x")
plt.ylabel("F(x)")
plt.legend()
plt.show()


#Use householder algorithm from exercises 3 to compute the least squares
#solution for the a_i coefficients
A_7 = householder_ls(LHS_Matrix(x,7), f)
print("The m= 7 LSS a_i coefficients are: " + str(A_7))

#Plot the polynomial just outside the range of the x values to see its 
#behaviour elsewhere 
X = np.linspace(-1.04,1.04,1000)
plt.figure(2)
plt.plot(X, PolyCoefficients(X, A_7))
plt.scatter(x,f,c='black')
plt.title("Approximation of F using the QR method and a polynomial of degree 7") 
plt.xlabel("x")
plt.ylabel("F(x)")
plt.show()


#Use householder algorithm from exercises 3 to compute the least squares
#solution for the a_i coefficients
A_7_perturbed = householder_ls(LHS_Matrix(x,7), f_perturbed)
print("The m= 7 LSS perturbed a_i coefficients are: " + str(A_7_perturbed))

#Plot the polynomial just outside the range of the x values to see its 
#behaviour elsewhere 
X = np.linspace(-1.04,1.04,1000)
plt.figure(3)
plt.plot(X, PolyCoefficients(X, A_7_perturbed), label = 'Perturbed F')
plt.plot(X, PolyCoefficients(X, A_7), label = 'F')
plt.title("Approximation of perturbed F using the QR method and a polynomial of degree 7") 
plt.xlabel("x")
plt.ylabel("F(x)")
plt.legend()
plt.show()


#Randomly generate 10 points and see how far away the 10 points are according
#to the perturbed function and unperturbed function, by using the inner product
#If the inner product is smaller, then the points are closer together, and so
#there is less sensitivity
#These results show that the polynomial of degree 10 is much more sensitive
np.random.seed(0)
X = np.random.uniform(-1,1,10)
X1 = PolyCoefficients(X, A_7_perturbed)-PolyCoefficients(X, A_7)
X2 = PolyCoefficients(X, A_10_perturbed)-PolyCoefficients(X, A_10)
error1 = np.inner(X1,X1)
error2 = np.inner(X2,X2)
print("The inner product of the difference between 10 points for m=7 is " + str(error1))
print("The inner product of the difference between 10 points for m=10 is " + str(error2))


#Here we compute the inner product of the difference between the perturbed and
#unperturbed coefficients to measure their relative sensitivity 
X3 = A_10_perturbed-A_10
X4 = A_7_perturbed-A_7
error1 = np.inner(X3,X3)
error2 = np.inner(X4,X4)
print("The inner product of the difference between the coefficients for m=10 is " + str(error1))
print("The inner product of the difference between the coefficients for m=7 is " + str(error2))