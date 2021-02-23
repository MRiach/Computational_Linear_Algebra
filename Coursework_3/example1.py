#import os
import numpy as np
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
from cla_utils import cond
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-mc-MRiach")
from polyfunctions import polyfitA
from polyfunctions import polyvalA
from polyfunctions import polyfit
from polyfunctions import polyval
import matplotlib.pyplot as plt


#points to be fitted and used in all plots, precomputed for efficiency
X0 = 2*np.arange(10001)/10000-1
X = np.cos(np.pi*np.arange(1001)/1000)
Y = 1*X
Y = 1+25*np.square(Y)
Y = np.reciprocal(Y)

#Vandermonde with Chebyshev points


plt.figure(0)
c2 = polyfit(X,Y,100)
Y1 = polyval(c2,X0)
plt.plot(X0,Y1)
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Approximation of f(x) without Vandermonde Arnoldi")
plt.show()

#Zoomed in
plt.figure(1)
plt.plot(X0[0:500],Y1[0:500])
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Approximation of f(x) without Vandermonde Arnoldi zoomed in")
plt.show()

#Vandermonde Arnoldi with Chebyshev points
plt.figure(2)
Q,H,d1 = polyfitA(X,Y,100)
Y2 = polyvalA(d1,H,X0)
plt.plot(X0,Y2)
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Approximation of f(x) with Vandermonde Arnoldi")
plt.show()


plt.figure(3)
plt.plot(X0[0:500],Y2[0:500])
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Approximation of f(x) with Vandermonde Arnoldi zoomed in")
plt.show()


#official solution at each of the points plotted used to compute the error
#for polynomials of varying degree with/without VA
Y3= 1*X0
Y3 = 1+25*np.square(Y3)
Y3 = np.reciprocal(Y3)

#condition numbers for A and Q and errors for Vandermonde/Arnoldi Vandermonde
cond0 = np.zeros(10)
cond1 = np.zeros(10)
errors0 = np.zeros(10)
errors1 = np.zeros(10)

for i in np.array(10*np.arange(1,11)):
    c2 = polyfit(X,Y,i)
    A = np.vander(X,int(i/10)+1,increasing = True)    
    Y1 = polyval(c2,X0)
    cond0[int(i/10-1)] = cond(A)
    errors0[int(i/10-1)] = np.linalg.norm(Y1-Y3)
    Q,H,d1 = polyfitA(X,Y,i)
    Y2 = polyvalA(d1,H,X0)
    cond1[int(i/10-1)] = cond(Q)
    errors1[int(i/10-1)] = np.linalg.norm(Y2-Y3)


plt.figure(4)    
plt.plot(np.array(range(1,11)), cond0,label='A')
plt.plot(np.array(range(1,11)), cond1,label='Q')
plt.legend()
plt.xlabel("degree of polynomial, n")
plt.xlim([1,10])
plt.ylabel("Condition number")
plt.yscale('log')
plt.title("Condition number vs degree of polynomial")
plt.show()

plt.figure(5)    
plt.plot(np.array(10*np.arange(1,11)), errors0,label='No Vandermonde Arnoldi')
plt.plot(np.array(10*np.arange(1,11)), errors1,label='Vandermonde Arnoldi')
plt.legend()
plt.xlabel("degree of polynomial, n")
plt.xlim([10,100])
plt.ylabel("Modulus of error")
plt.yscale('log')
plt.title("Error vs degree of polynomial of 10k plotted points")
plt.show()