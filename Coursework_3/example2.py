#import os
import numpy as np
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-mc-MRiach")
from polyfunctions import polyfitA
from polyfunctions import polyvalA
from polyfunctions import polyfit
from polyfunctions import polyval
import matplotlib.pyplot as plt

#points to be fitted and used in all plots, precomputed for efficiency
U1 = np.linspace(-1,-1/3,500)
U2 = np.linspace(1/3,1,500)
X = np.concatenate([U1,U2])
Y = np.sign(X)
X0 = np.linspace(-1.1,1.1,10000)


#Vandermonde with Chebyshev points


plt.figure(0)
c2 = polyfit(X,Y,75)
Y1 = polyval(c2,X0)
plt.plot(X0,Y1)
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Approximation of f(x) with Vandermonde")
plt.ylim([-1.5,1.5])
plt.show()


#Vandermonde Arnoldi with Chebyshev points
plt.figure(1)
Q,H,d1 = polyfitA(X,Y,75)
Y2 = polyvalA(d1,H,X0)
plt.plot(X0,Y2)
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Approximation of f(x) with Vandermonde Arnoldi")
plt.ylim([-1.5,1.5])
plt.show()



#official solution at each of the points plotted used to compute the error
#for polynomials of varying degree with/without VA
X0 = np.linspace(-0.9,-0.3,10000)
Y3= 1*X0
Y3 = np.sign(X0)


#errors for Vandermonde/Arnoldi Vandermonde
errors0 = np.zeros(10)
errors1 = np.zeros(10)

for i in np.array(10*np.arange(1,11)):
    c2 = polyfit(X,Y,i)   
    Y1 = polyval(c2,X0)
    errors0[int(i/10-1)] = np.linalg.norm(Y1-Y3)
    Q,H,d1 = polyfitA(X,Y,i)
    Y2 = polyvalA(d1,H,X0)
    errors1[int(i/10-1)] = np.linalg.norm(Y2-Y3)




plt.figure(2)    
plt.plot(np.array(10*np.arange(1,11)), errors0,label='Vandermonde')
plt.plot(np.array(10*np.arange(1,11)), errors1,label='Vandermonde Arnoldi')
plt.legend()
plt.xlabel("degree of polynomial, n")
plt.xlim([10,100])
plt.ylabel("Modulus of error")
plt.yscale('log')
plt.title("Error vs degree of polynomial of 10k plotted points")
plt.show()