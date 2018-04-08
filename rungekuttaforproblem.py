from nodepy import rk
import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
import pylab
#import nodepy.linear_multistep_method as lm
import ivp4
from ivp4 import detest
rk44 = rk.loadRKM('RK44')
#ab4=lm.Adams_Bashforth(4)
ivp=ivp4
myivp=ivp4.detest('2OD')
#t,u = ab4(myivp)
t,u = rk44(myivp)
print(t,u)
t1 = np.array([0.0, 1.0, 2.0, 3.0]) 
z1 = np.array([100.0, 110.0, 70.0,73.0])
z2 = np.array([70.0, 50.0, 10.0,52.0])#for j in jlist
plt.axis([0, 5,0,120])
plt.plot(t, u, "b-", )#model solution
plt.plot(t1, z1, "r--",marker='o' )#fact value
plt.plot(t1, z2, "g--",marker='o' )#fact value

plt.xlabel('t')
plt.ylabel('u')
plt.legend(['x(t)', 'y(t)','z1(t)','z2(t)'], loc = 'central')
plt.grid()
plt.show() 
