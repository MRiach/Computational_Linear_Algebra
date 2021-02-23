#import os
import numpy as np
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
from cla_utils import householder_ls
from cla_utils import cond
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-mc-MRiach")
from polyfunctions import polyfitA
from polyfunctions import polyvalA
from polyfunctions import polyfit
from polyfunctions import polyval
import matplotlib.pyplot as plt

#Vandermonde with uniformly spaced out points
X = 2*np.arange(1001)/1000-1
Y = 1*X
Y = 1+30*np.square(Y)
Y = np.reciprocal(Y)


plt.figure(0)
c1 = polyfit(X,Y,300)
X0 = 2*np.arange(10001)/10000-1
Y0 = polyval(c1,X0)
plt.plot(X0,Y0)
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Approximation of f(x) using equidistant points")
plt.show()

#Zoomed in
plt.figure(1)
plt.plot(X0[0:500],Y0[0:500])
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Approximation of f(x) using equidistant points zoomed in")
plt.show()




#Vandermonde with Chebyshev points
Z = np.cos(np.pi*np.arange(1001)/1000)
Y = 1*Z
Y = 1+30*np.square(Y)
Y = np.reciprocal(Y)


plt.figure(2)
c2 = polyfit(Z,Y,300)
Y3 = polyval(c2,X0)
plt.plot(X0,Y3)
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Approximation of f(x) using Chebyshev points")
plt.show()

#Zoomed in
plt.figure(3)
plt.plot(X0[0:500],Y3[0:500])
plt.xlabel("x")
plt.ylabel("p(x)")
plt.title("Approximation of f(x) using Chebyshev points zoomed in")
plt.show()

