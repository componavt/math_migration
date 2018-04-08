
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dz/dt
def model(z,t):
    x = z[0]
    y = z[1]
 #system of differential equations   
    dxdt = 40.0+((10.0-15.0)/100.0)*x-(35.0/100.0)*x+((20.0-10.0)/(100.0*70.0))*x*y
    dydt =20.0+((10.0-25.0)/70.0)*y-(15.0/70.0)*y+((10.0-20.0)/(70.0*100.0))*x*y 
    dzdt=[dxdt,dydt]
 
    
    return dzdt
    

z0 = [100.0,70.0]# initial condition


t = np.linspace(0,10)# time points for graph of solution
z = odeint(model,z0,t)# solve ODE
t1 = np.array([0.0, 1.0, 2.0, 3.0])# time of fact values of population in regions 
z1 = np.array([100.0, 110.0, 70.0,73.0])#fact values for number of population in 1 region in time t1
z2 = np.array([70.0, 50.0, 10.0,52.0])#fact values for number of population in 2 region in time t1
# plot results
plt.plot(t,z[:,0],'b-',label=r'X(t)')# graph of fact value of number of population in 1 region  
plt.plot(t,z[:,1],'g-',label=r'Y(t)')# graph of fact value of number of population in 2 region 
plt.plot(t1, z1, "b--",marker='o' , label='z1')# graph of fact value of number of population in 1 region  
plt.plot(t1, z2, "g--",marker='o',label='z2')#fact value of number of population in 1 regionplt.ylabel('response')
plt.xlabel('time')
plt.legend(loc='under')
plt.show()
