from nodepy import rk
import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
import pylab
import ivp6
from ivp6 import detest
rk44 = rk.loadRKM('RK44')#load runge-kutta
#ivp=ivp4
myivp=ivp6.detest('2OD')#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
print(t,u)
from ivp6 import u0,u1,u2,u3
t1 = np.array([0.0, 1.0, 2.0, 3.0])#time for fact value
z1 = np.array([u0[0], u1[0], u2[0],u3[0]])#fact value
z2 = np.array([u0[1], u1[1], u2[1],u3[1]])#fact value
plt.axis([0, 5,0,120])
plt.plot(t, u, "b-", )#model solution
plt.plot(t1, z1, "r--",marker='o' )#fact value
plt.plot(t1, z2, "g--",marker='o' )#fact value

plt.xlabel('t')
plt.ylabel('u')
plt.legend(['x(t)', 'y(t)','z1(t)','z2(t)'], loc = 'central')
plt.grid()
plt.show() 
