from nodepy import rk
import numpy as np
from matplotlib import mlab
import matplotlib.pyplot as plt
import pylab
import ivptest
from ivptest import detest
rk44 = rk.loadRKM('RK44')#load runge-kutta
ivp=ivptest
myivp=ivptest.detest('2OD1')#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem

plt.axis([0, 10,0,20])
plt.plot(t, u,)#model solution
plt.xlabel('t')
plt.ylabel('u')
plt.legend(['x(t)', 'y(t)'], loc = 'central')
plt.grid()
plt.show() 
