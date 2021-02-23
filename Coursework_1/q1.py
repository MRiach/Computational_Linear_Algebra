#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This year's modules\\CLA\\comp-lim-alg-course")
import numpy as np
from cla_utils import *
import matplotlib.pyplot as plt
from matplotlib import colors
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This year's modules\\CLA\\comp-lim-alg-course\\clacourse-2020-cw1-MRiach")

A = np.load('values.npy')
m,n = A.shape

#QR factorisation using householder, since CGS and MGS return reduced QR
#and householder yields a more numerically stable and accurate result
Q, R = householder_qr(A)


#Reduced QR extracted from original QR 
Q_hat, R_hat = Q[:,0:100], R[0:100,0:100]


#Plot the trajectory of random functions in A 
plt.figure(0)
plt.title("Trajectories of rows of A") 
plt.xlabel("Time, t")
plt.ylabel("Value at Time, t")
for i in range(m):
 plt.plot(np.array(range(n)), A[i, :])
  
plt.show()


#Unhashtag the below lines of code to export Q_hat and R_hat to a csv file
#np.savetxt('Q_hat.csv', Q_hat, delimiter=',')
#np.savetxt('R_hat.csv', R_hat, delimiter=',')

#Function to plot heat map
def heatmap2d(arr: np.ndarray):
    norm = colors.LogNorm(10**-25,10**3)
    plt.imshow(arr, cmap='viridis',norm=norm)
    plt.colorbar()
    plt.title("Heat map of R_hat") 
    plt.show()

#Plot of heat map 
heatmap2d(R_hat)

#Plot of histograms
plt.figure(1)
plt.hist(Q_hat[:,0])
plt.title("Histogram of 1st row of Q_hat transposed") 
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.figure(2)
plt.hist(Q_hat[:,49])
plt.title("Histogram of 50th row of Q_hat transposed") 
plt.xlabel("Value")
plt.ylabel("Frequency")