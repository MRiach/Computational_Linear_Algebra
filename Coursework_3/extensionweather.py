#import os
import numpy as np
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\comp-lim-alg-course")
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-mc-MRiach")
from polyfunctions import polyfitA
from polyfunctions import polyvalA
from polyfunctions import polyfit
from polyfunctions import polyval
import matplotlib.pyplot as plt


#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This years modules\\CLA\\cla-2020-mc-MRiach")
#import data from csv file and tidy it up
weather_data = np.genfromtxt('weatherdata.csv', delimiter=',')
weather_data = np.delete(weather_data,0,1)
weather_data = np.delete(weather_data,0,0)

#plot original data
plt.figure(0)
plt.plot(weather_data[:,1],weather_data[:,0])
plt.xlabel("Time from 1st January 2021 midnight (hours) ")
plt.ylabel("Temperature (Celcius)")
plt.title("Temperature in London over time")
plt.show()

#official solution at each of the points plotted used to compute the error
#for polynomials of varying degree with/without VA
X0 = weather_data[:,1]
Y3= weather_data[:,0]


#pick every other point as data point
X = X0[::2]
Y = Y3[::2]


#Vandermonde
plt.figure(0)
c2 = polyfit(X,Y,25)
Y1 = polyval(c2,X0)
plt.plot(X0,Y1)
plt.xlabel("Time from 1st January 2021 midnight (hours)")
plt.ylabel("Temperature (Celcius)")
plt.title("Approximation of Temperatures with Vandermonde")
plt.show()


#Vandermonde Arnoldi
plt.figure(1)
Q,H,d1 = polyfitA(X,Y,25)
Y2 = polyvalA(d1,H,X0)
plt.plot(X0,Y2)
plt.xlabel("Time from 1st January 2021 midnight (hours)")
plt.ylabel("Temperature (Celcius)")
plt.title("Approximation of Temperatures with Vandermonde Arnoldi")
plt.show()

errors0 = np.zeros(10)
errors1 = np.zeros(10)

for i in np.array(3*np.arange(1,11)):
    c2 = polyfit(X,Y,i)   
    Y1 = polyval(c2,X0)
    errors0[int(i/3-1)] = np.linalg.norm(Y1-Y3)
    Q,H,d1 = polyfitA(X,Y,i)
    Y2 = polyvalA(d1,H,X0)
    errors1[int(i/3-1)] = np.linalg.norm(Y2-Y3)
    
plt.figure(2)    
plt.plot(np.array(3*np.arange(1,11)), errors0,label='Vandermonde')
plt.plot(np.array(3*np.arange(1,11)), errors1,label='Vandermonde Arnoldi')
plt.legend()
plt.xlabel("degree of polynomial, n")
plt.xlim([3,30])
plt.ylabel("Modulus of error")
plt.title("Error vs degree of polynomial plotted at hourly intervals")
plt.show()