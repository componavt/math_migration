from math import exp
import numpy as np
from numpy import *
import numpy.linalg as ln
from matplotlib import mlab
import matplotlib.pyplot as plt
import pylab
import nodepy
from nodepy import rk
eps1 = 0.1#accuracy
mu0=10**4 #value for changing the Hessian matrix
a = np.array([20.,10.,20. ])#arrive to region,when time is 0,1,2 in 1 and 2 region respectively between 1 b 2 region
l=np.array([10.,50.,20.])#leave region, time is 0,1,2 from 1 and 2 region respectively between 1 b 2


u0=100.#initial value of population in 1 and 2 region respectively,third parameter is constant population abroad 
u1= (u0+a[0]-l[0])#numer of population, time 1
u2= (u1+a[1]-l[1])#numer of population, time 2
u3= (u2+a[2]-l[2])#numer of population, time 3

aa1=((a[0]/u0)+(a[1]/u1)+(a[2]/u2))/3.0#average number of population, who arrives to region 1
#aa2=((a[1]/u0[1])+(a[3]/u1[1])+(a[5]/u2[1]))/3.0#average number of population, who arrives to region 2
#print('Average number of population, who arrives to region 1 is', aa1)
al1=((l[0]/u0)+(l[1]/u1)+(l[2]/u2))/3.0#average number of population, who leaves region 1
#al2=((l[1]/u0[1])+(l[3]/u1[1])+(l[5]/u2[1]))/3.0#average number of population, who leaves region 2
#print('Average number of population, who leaves region 1 is', al1)
s0= np.array([aa1,al1])#vector of initial parameters
dq=0.00001
funclist=[ ]
t = np.linspace(0.,3.)
delta1=1#weight coefficient for function, which must be minimized


class IVPOPTIM(object):  #Problems that are solved by the Runge-Kutta method
     def __init__(self, f=None, u0=100., t0=0., T=3.0, dt0=0.01, exact=None, desc='', name=''):
        self.u0  = u0
        self.rhs = f
        self.T   = T
        self.exact = exact
        self.description = desc
        self.t0 = t0
        self.dt0 = 0.01 
        self.name = name
     def __repr__(self):
           return 'Problem Name:  '+self.name+'\n'+'Description:   '+self.description   
def detest(testkey):#Description of problem
      import numpy as np
      ivpoptim=IVPOPTIM()
      if testkey=='2OD':#system of differential equations        
        ivpoptim.u0=u0#initial parameters
        ivpoptim.T=3.0#Final time of integration
        ivpoptim.rhs =lambda t,u: (s0[0]*u-s0[1]*u)  #right side of ODE system
        ivpoptim.dt0 = 0.01#time step
      else: raise Exception('Unknown Detest problem')
      ivpoptim.name=testkey
      ivpoptim.description='Problem '+testkey+' of the non-stiff DETEST suite.'
      return ivpoptim  

rk44 = rk.loadRKM('RK44')#load runge-kutta
ivp=detest('2OD')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem   
u4=np.array([u[1667], u[3334], u[5001]])
t4=np.array([t[1667], t[3334], t[5001]])
#print(u4)
pogr1=([(u4[0]-u1),(u4[1]-u2),(u4[2]-u3)])#difference between model and fact values when time is 1,2,3
norma1=round(ln.norm(pogr1[0]),4)#the distance between the points of the model and fact values when time is 1
norma2=round(ln.norm(pogr1[1]),4)#the distance between the points of the model and fact values when time is 2
norma3=round(ln.norm(pogr1[2]),4)#the distance between the points of the model and fact values when time is 3
print('the distance between the points of the model and fact values when time is 1',  norma1)
print('the distance between the points of the model and fact values when time is 2', norma2)
print('the distance between the points of the model and fact values when time is 1',  norma3)
norma=np.array([norma1,norma2,norma3])
maxnorma=max(norma)
print('the maximum distance between the points of the model and fact values',maxnorma)               
t1 = np.array([0.0, 1.0, 2.0, 3.0])#time for fact value
z1 = np.array([u0, u1, u2,u3])#fact value

plt.axis([0, 5,0,200])
plt.plot(t, u, )#model solution
plt.plot(t1, z1, "r--",marker='o' )#fact value


plt.xlabel('t')
plt.ylabel('u')
plt.legend(['x(t)','z1(t)'], loc = 'central')
plt.grid()
plt.show()















