import numpy as np
import matplotlib.pyplot as plt


def grad(x,y):
    m_curr=b_curr=0
    iterations=10
    N=len(x)
    plt.scatter(x,y,color='red',marker='o',linewidths=5)
    learning_rate=0.01
    for i in range(iterations):
        yp=m_curr*x+b_curr
        plt.plot(x,yp,color='blue')
        cost=sum((y-yp)**2)*(1/N)
        dm=(2/N)* sum(x*(y-yp))
        db=(2/N)* sum((y-yp))

        m_curr=m_curr - learning_rate + dm
        b_curr=b_curr - learning_rate + db

        print("m {}, b {},cost {} iteration {}".format(m_curr,b_curr,cost,i))

    pass

x=np.array([1,2,3,4,5])
y=np.array([6,7,9,11,15])

grad(x,y)
plt.show()