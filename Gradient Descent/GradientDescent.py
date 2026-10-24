#Gradient Descent for Linear Regression
#ypred = mx + b
#loss=(y-ypred)**2/N

import numpy as np
import matplotlib.pyplot as plt
#Initialise some parameters
x = np.random.randn(10,1)
y = 2*x + np.random.randn()
#Parameters
m=0
b=0
#Hyperparameter
learning_rate=0.01


#Create gradient descent function
def descent(x,y,m,b,learning_rate):
    dldb=0
    dldm=0
    N = x.shape[0]
    #loss=(y-(mx+b))**2
    for xi,yi in zip(x,y):
        dldm += (-2*xi*(yi-(m*xi+b)))/N
        dldb += (-2 * (yi - (m * xi + b)))/N

        #Make an update to the m
        m = m - learning_rate *  dldm
        b = b - learning_rate *  dldb

        return m,b

#Iteratively Make Updates
for epoch in range(400):
    m,b=descent(x,y,m,b,learning_rate)
    ypred=m*x + b
    loss=np.divide(sum((y-ypred)**2), x.shape[0])
    print(f'{epoch} loss is {loss}, parameters m:{m} b:{b}')
    print(m,b)
    break
    #Run Gradient Descent
print(x,y)
plt.plot(m,loss)
plt.show()















