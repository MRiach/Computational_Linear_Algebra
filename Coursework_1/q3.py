#import os 
#os.chdir("C:\\Users\\Marwan Riach\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This year's modules\\CLA\\comp-lim-alg-course")
#os.chdir("C:\\Users\\marwa\\OneDrive\\Documents\\Fourth Year Maths Imperial\\This year's modules\\CLA\\comp-lim-alg-course")
import numpy as np 
from cla_utils import *
import matplotlib.pyplot as plt

def get_V_Matrix(p,M):
    """
    Creates V matrix as given in q3
    """       
    V = np.zeros([2*M,p+1])
    
    x = np.linspace(-M,M-1,2*M)
    
    x = (x+0.5)/M

    for i in range(p+1):
       V[:,i] = np.power(x,i)
    
    return V 

#Get the Q and R as in the report
Q,R = householder_qr(get_V_Matrix(5,100))
Q = Q[:,:6]
R = R[:6,:]

#plot of columns of Q
plt.figure(0)
plt.ylim(-0.3,0.5)
plt.plot(Q[:,0],label = 'Column 1')
plt.plot(Q[:,1],label = 'Column 2')
plt.plot(Q[:,2],label = 'Column 3')
plt.plot(Q[:,3],label = 'Column 4')
plt.plot(Q[:,4],label = 'Column 5')
plt.plot(Q[:,5],label = 'Column 6')
plt.title("Plot of columns of Q") 
plt.xlabel("Row number")
plt.ylabel("Value in corresponding row number")
plt.legend(loc='upper left')
plt.show()


#Plots for 4b
A = get_V_Matrix(100,100)

Q,R = householder_qr(A)
Q = Q[:,:101]
R = R[:101,:]
plt.figure(1)
#Plot last 5 columns
plt.plot(Q[:,-1])
plt.plot(Q[:,-2])
plt.plot(Q[:,-3])
plt.plot(Q[:,-4])
plt.plot(Q[:,-5])
plt.title("Plot of last 5 columns of Q with Householder") 
plt.xlabel("Row number")
plt.ylabel("Value in corresponding row number")


Q1,R1 = GS_classical(A)
plt.figure(2)
#Plot last 5 columns
plt.plot(Q[:,-1])
plt.plot(Q[:,-2])
plt.plot(Q[:,-3])
plt.plot(Q[:,-4])
plt.plot(Q[:,-5])
plt.title("Plot of last 5 columns of Q with CGS") 
plt.xlabel("Row number")
plt.ylabel("Value in corresponding row number")


Q2,R2 = GS_modified(A)
plt.figure(3)
#Plot last 5 columns
plt.plot(Q[:,-1])
plt.plot(Q[:,-2])
plt.plot(Q[:,-3])
plt.plot(Q[:,-4])
plt.plot(Q[:,-5])
plt.title("Plot of last 5 columns of Q with MGS") 
plt.xlabel("Row number")
plt.ylabel("Value in corresponding row number")

#Output of orthogonality error for last five columns
error0 = np.linalg.norm(np.dot(Q[:,-5:].T,Q[:,-5:])-np.eye(5),ord=np.inf)
error1 = np.linalg.norm(np.dot(Q1[:,-5:].T,Q1[:,-5:])-np.eye(5),ord=np.inf)
error2 = np.linalg.norm(np.dot(Q2[:,-5:].T,Q2[:,-5:])-np.eye(5),ord=np.inf)

print("The orthogonality error for Householder is " + str(error0))
print("The orthogonality error for CGS is " + str(error1))
print("The orthogonality error for MGS is " + str(error2))