#!/usr/bin/python
# -*- coding: <ASCII> -*-
import os, sys
from math import exp
import numpy as np
from numpy import *
import numpy.linalg as ln
import nodepy
from nodepy import rk
import bigfloat
from decimal import Decimal
from matplotlib import mlab
import matplotlib.pyplot as plt
#TWOPLACES = Decimal(10) ** -8
eps1 = 0.1#accuracy
mu0=10**6 #value for changing the Hessian matrix

u0=np.array([105.276,37.629])#initial value of population in 1 and 2 region respectively
u1= np.array([105.241,37.624])#numer of population, time 1
u2= np.array([105.386,37.67])#numer of population, time 2
u3= np.array([105.619,37.728])#numer of population, time 3
u4= np.array([105.914,37.753])#numer of population, time 4
u5= np.array([106.173,37.799])#numer of population, time 5 
ulist1=([u0[0], u1[0], u2[0],u3[0],u4[0],u5[0]])
ulist2=([u0[1], u1[1], u2[1],u3[1],u4[1],u5[1]])
v=np.array([0.72,0.79])# true utility of 1 region, 2 regions 
k1=0.000436667#average value of natural growth in 1 region
k2=0.001211111 #average value of natural growth in 2 region

beta1=0.86 #parameter betta1    
beta2=0.84#parameter betta2 
dq=10**(-6)
q=50#positive parameter 
funclist=[ ]
t = np.linspace(0.,6.)
delta1=0.7
#weight coefficient for function, which must be minimized,delta1+delta2=1
delta2=0.3
 #weight coefficient for function, which must be minimized,delta1+delta2=1 
s00=0.001526164  #coefficient of migration  from 1 region to 2
s01=0.007097008  #coefficient of migration  from 2 region to 1



s0=np.array([s00,s01,beta1,beta2,k1,k2])#vector of initial parameters      
s0list=[s0]

class IVPOPTIM(object):  #Problems that are solved by the Runge-Kutta method
     def __init__(self, f=None,u0=np.array([105.276,37.629]) , t0=0., T=6.0, dt0=0.001, exact=None, desc='', name='',dt=0.001):
        self.u0  = u0
        self.rhs = f
        self.T   = T
        self.exact = exact
        self.description = desc
        self.t0 = t0
        self.dt0 = 0.001
        self.dt=0.001
        self.name = name
     def __repr__(self):
           return 'Problem Name:  '+self.name+'\n'+'Description:   '+self.description

def detest(testkey):#Description of problem
      import numpy as np
      ivpoptim=IVPOPTIM()
      if testkey=='2OD':#system of differential equations        
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=6.0#Final time of integration     

         ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                          exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                       exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
         ivpoptim.dt0 = 0.001#time step
         ivpoptim.dt = 0.001

      elif testkey=='2OD1': # a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration

            ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.001#time step
            ivpoptim.dt = 0.001
      elif testkey=='2OD2':# a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration 

            ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.001#time step
            ivpoptim.dt = 0.001
      elif testkey=='2OD3':#system of differential equations        
             ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
             ivpoptim.T=6.0#Final time of integration     

             ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1]), 
                                          (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1])]) 

             ivpoptim.dt0 = 0.001#time step
             ivpoptim.dt = 0.001
      elif testkey=='2OD4':#system of differential equations        
             ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
             ivpoptim.T=6.0#Final time of integration     

             ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
             ivpoptim.dt0 = 0.001#time step
             ivpoptim.dt = 0.001
      elif testkey=='2OD11':# a system of differential equations with the first parameter changed by an amount dq
           ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
           ivpoptim.T=6.0#Final time of integration

           ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-(s0[0]+2*dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+(s0[0]+2*dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
           ivpoptim.dt0 = 0.001#time step
           ivpoptim.dt = 0.001
      elif testkey=='2OD12':# a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration

            ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.001#time step
            
            ivpoptim.dt = 0.001
      elif testkey=='2OD13':# a system of differential equations with the first parameter changed by an amount dq
           ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
           ivpoptim.T=6.0#Final time of integration

           ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1])]) 
           ivpoptim.dt0 = 0.001#time step
           ivpoptim.dt = 0.001
      elif testkey=='2OD14':# a system of differential equations with the first parameter changed by an amount dq
           ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
           ivpoptim.T=6.0#Final time of integration

           ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-(s0[0]+dq)*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+(s0[0]+dq)*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
           ivpoptim.dt0 = 0.001#time step
           ivpoptim.dt = 0.001

      elif testkey=='2OD22':# a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration

            ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+(s0[1]+2*dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]),
                                                 (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-(s0[1]+2*dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.001#time step
            ivpoptim.dt = 0.001
      elif testkey=='2OD23':# a system of differential equations with the first parameter changed by an amount dq
          ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
          ivpoptim.T=6.0#Final time of integration

          ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]+(s0[1]+dq)*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]-(s0[1]+dq)*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1])]) 
          ivpoptim.dt0 = 0.001#time step
          ivpoptim.dt = 0.001

      elif testkey=='2OD24':# a system of differential equations with the first parameter changed by an amount dq
          ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
          ivpoptim.T=6.0#Final time of integration

          ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
          ivpoptim.dt0 = 0.001#time step
          ivpoptim.dt = 0.001

      elif testkey=='2OD33':#system of differential equations        
          ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
          ivpoptim.T=6.0#Final time of integration     

          ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0])))))*u[1])]) 
          ivpoptim.dt0 = 0.001#time step
          ivpoptim.dt = 0.001

      elif testkey=='2OD34':#system of differential equations        
          ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
          ivpoptim.T=6.0#Final time of integration     

          ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1])]) 
          ivpoptim.dt0 = 0.001#time step   
          ivpoptim.dt = 0.001

      elif testkey=='2OD44':#system of differential equations        
          ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
          ivpoptim.T=6.0#Final time of integration     

          ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
          ivpoptim.dt0 = 0.001#time step
          ivpoptim.dt = 0.001
      elif testkey=='2OD5':#system of differential equations        
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=6.0#Final time of integration     

         ivpoptim.rhs = lambda t,u: np.array([((s0[4]+dq)*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                          exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                       exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
         ivpoptim.dt0 = 0.001#time step
         ivpoptim.dt = 0.001


      elif testkey=='2OD6':#system of differential equations        
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=6.0#Final time of integration     

         ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                          exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           ((s0[5]+dq)*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                       exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
         ivpoptim.dt0 = 0.001#time step
         ivpoptim.dt = 0.001



      elif testkey=='2OD51': # a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration

            ivpoptim.rhs = lambda t,u: np.array([((s0[4]+dq)*u[0]-(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.001#time step
            ivpoptim.dt = 0.001
      elif testkey=='2OD52':# a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration 

            ivpoptim.rhs = lambda t,u: np.array([((s0[4]+dq)*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.001#time step
            ivpoptim.dt = 0.001
      elif testkey=='2OD53':#system of differential equations        
             ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
             ivpoptim.T=6.0#Final time of integration     

             ivpoptim.rhs = lambda t,u: np.array([((s0[4]+dq)*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1]), 
                                          (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1])]) 

             ivpoptim.dt0 = 0.001#time step
             ivpoptim.dt = 0.001
      elif testkey=='2OD54':#system of differential equations        
             ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
             ivpoptim.T=6.0#Final time of integration     

             ivpoptim.rhs = lambda t,u: np.array([((s0[4]+dq)*u[0]-s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
             ivpoptim.dt0 = 0.001#time step
             ivpoptim.dt = 0.001

      

      elif testkey=='2OD55':#system of differential equations        
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=6.0#Final time of integration     

         ivpoptim.rhs = lambda t,u: np.array([((s0[4]+2*dq)*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                          exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (s0[5]*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                       exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
         ivpoptim.dt0 = 0.001#time step
         ivpoptim.dt = 0.001
     


      elif testkey=='2OD6':#system of differential equations        
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=6.0#Final time of integration     

         ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                          exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           ((s0[5]+dq)*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                       exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
         ivpoptim.dt0 = 0.001#time step
         ivpoptim.dt = 0.001

      elif testkey=='2OD61': # a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration

            ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           ((s0[5]+dq)*u[1]+(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.001#time step
            ivpoptim.dt = 0.001
      elif testkey=='2OD62':# a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration 

            ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           ((s0[5]+dq)*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.001#time step
            ivpoptim.dt = 0.001
      elif testkey=='2OD63':#system of differential equations        
             ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
             ivpoptim.T=6.0#Final time of integration     

             ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1]), 
                                          ((s0[5]+dq)*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1])]) 

             ivpoptim.dt0 = 0.001#time step
             ivpoptim.dt = 0.001
      elif testkey=='2OD64':#system of differential equations        
             ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
             ivpoptim.T=6.0#Final time of integration     

             ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           ((s0[5]+dq)*u[1]+s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
             ivpoptim.dt0 = 0.001#time step
             ivpoptim.dt = 0.001


      elif testkey=='2OD65':#system of differential equations        
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=6.0#Final time of integration     

         ivpoptim.rhs = lambda t,u: np.array([((s0[4]+dq)*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                          exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           ((s0[5]+dq)*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                       exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
         ivpoptim.dt0 = 0.001#time step
         ivpoptim.dt = 0.001

      elif testkey=='2OD66':#system of differential equations        
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=6.0#Final time of integration     

         ivpoptim.rhs = lambda t,u: np.array([(s0[4]*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                          exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           ((s0[5]+2*dq)*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                       exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
         ivpoptim.dt0 = 0.001#time step
         ivpoptim.dt = 0.001


      else: raise Exception('Unknown Detest problem')
      ivpoptim.name=testkey
      ivpoptim.description='Problem '+testkey+' of the non-stiff DETEST suite.'
      return ivpoptim

      


rk44 = rk.loadRKM('RK44')#load runge-kutta
ivp=detest('2OD')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
r=abs(t[1]-t[0])#the size of step
r11=int(1/r)
r21=int(2/r)
r31=int(3/r)
r41=int(4/r)
r51=int(5/r)

r12=int(1/r)+1
r22=int(2/r)+1
r32=int(3/r)+1
r42=int(4/r)+1
r52=int(5/r)+1
print(r11,r12, r21,r22,r31,r32,r41,r42,r51,r52)
print(t[r11],t[r21],t[r31],t[r41],t[r51],t[r12],t[r22],t[r32],t[r42],t[r52])
if abs(1-t[r11])<abs(1-t[r12]):     
     r1=r11
else: r1=r12
print(abs(1-t[r11]),abs(1-t[r12]))
print(r1)

if abs(2-t[r21])<abs(2-t[r22]):     
     r2=r21
else: r2=r22
print(abs(2-t[r21]),abs(2-t[r22]))
print(r2)

if abs(3-t[r31])<abs(3-t[r32]):     
     r3=r31
else: r3=r32
print(abs(3-t[r31]),abs(3-t[r32]))
print(r3)

if abs(4-t[r41])<abs(4-t[r42]):     
     r4=r41
else: r4=r42
print(abs(4-t[r41]),abs(4-t[r42]))
print(r4)

if abs(5-t[r51])<abs(5-t[r52]):     
     r5=r51
else: r5=r52
print(abs(5-t[r51]),abs(5-t[r52]))
print(r5)
print(t[r1], t[r2], t[r3],t[r4], t[r5])

#t567=(t[r1]+t[834])/2
#print(t567)
u6=np.array([u[r1],u[r2],u[r3],u[r4],u[r5]])#approximate soluton in moment of time 1,2,3,4,5

ivp=detest('2OD1')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
u7=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])


ivp=detest('2OD2')
myivp=ivp  #load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
u8=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

ivp=detest('2OD3')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
u9=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

ivp=detest('2OD4')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
u40=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])
ivp=detest('2OD5')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
u50=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

ivp=detest('2OD6')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
u60=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

x0=np.array([u6[0][0],u6[1][0],u6[2][0],u6[3][0],u6[4][0],u6[0][1],u6[1][1],u6[2][1],u6[3][1],u6[4][1]])#array of solution in moment of time 1,2,3
xfact=np.array([u1[0],u2[0],u3[0],u4[0],u5[0],u1[1],u2[1],u3[1],u4[1],u5[1]])#fact values in moment of time 1,2,3
x1=np.array([u7[0][0],u7[1][0],u7[2][0],u7[3][0],u7[4][0],u7[0][1],u7[1][1],u7[2][1],u7[3][1],u7[4][1]])#solutions with with the first parameter changed by an amount dq 
x2=np.array([u8[0][0],u8[1][0],u8[2][0],u8[3][0],u8[4][0],u8[0][1],u8[1][1],u8[2][1],u8[3][1],u8[4][1]])#solutions with with the second parameter changed by an amount dq
x3=np.array([u9[0][0],u9[1][0],u9[2][0],u9[3][0],u9[4][0],u9[0][1],u9[1][1],u9[2][1],u9[3][1],u9[4][1]])#solutions with with the third parameter changed by an amount dq
x4=np.array([u40[0][0],u40[1][0],u40[2][0],u40[3][0],u40[4][0], u40[0][1],u40[1][1],u40[2][1],u40[3][1],u40[4][1]])#solutions with with the 4 parameter changed by an amount dq
x5=np.array([u50[0][0],u50[1][0],u50[2][0],u50[3][0],u50[4][0],u50[0][1],u50[1][1],u50[2][1],u50[3][1],u50[4][1]])#solutions with with the third parameter changed by an amount dq
x6=np.array([u60[0][0],u60[1][0],u60[2][0],u60[3][0],u60[4][0], u60[0][1],u60[1][1],u60[2][1],u60[3][1],u60[4][1]])#so

x01list1=([u0[0],u6[0][0],u6[1][0],u6[2][0],u6[3][0],u6[4][0]])
x01list2=([u0[1],u6[0][1],u6[1][1],u6[2][1],u6[3][1],u6[4][1]])

func=delta1*sum((xfact[0:5]-x0[0:5])**2)+delta2*sum((xfact[5:]-x0[5:])**2)#function which must be minimized
funclist.append(func)#value of a function with initial parameters
i=0 #element number in the array of value of a functions
pogr1=abs(x0[0]-xfact[0])#error in 1 time  for 1 region
pogr2=abs(x0[1]-xfact[1])#error in 2 time for 1 region
pogr3=abs(x0[2]-xfact[2] )#error in 3 time for 1 region
pogr4=abs(x0[3]-xfact[3])#error in 4 time for 1 region
pogr5=abs(x0[4]-xfact[4])#error in 5 time for 1 region

pogr6=abs(x0[5]-xfact[5])#error in 1 time for 2 region
pogr7=abs(x0[6]-xfact[6])#error in 2 time for 2 region
pogr8=abs(x0[7]-xfact[7])#error in 3 time  for 2 region
pogr9=abs(x0[8]-xfact[8])#error in 4 time for 2 region
pogr10=abs(x0[9]-xfact[9])#error in 5 time for 2 region



vectpogr1=[pogr1,pogr2,pogr3,pogr4,pogr5]#vector of error for 1 region
totalpogr1=ln.norm(vectpogr1)#total error for 1 region
totalpogrlist1=[totalpogr1]# for plot of total error for 1 region

vectpogr2=[pogr6,pogr7,pogr8,pogr9,pogr10]#vector of error for 2 region
totalpogr2=ln.norm(vectpogr2)#total error for 2 region
totalpogrlist2=[totalpogr2]# for plot of total error for 2 region
print('value for changing the Hessian matrix in 0 step:')
print(mu0)
print('value of function which must be minimized in 0 step:')
print(func)                   
v1=np.array([((x1[0]-x0[0])/dq), ((x1[1]-x0[1])/dq),((x1[2]-x0[2])/dq), ((x1[3]-x0[3])/dq),((x1[4]-x0[4])/dq),
             ((x1[5]-x0[5])/dq),((x1[6]-x0[6])/dq), ((x1[7]-x0[7])/dq),((x1[8]-x0[8])/dq), ((x1[9]-x0[9])/dq)])#derivative with the first parameter
v2=np.array([((x2[0]-x0[0])/dq), ((x2[1]-x0[1])/dq),((x2[2]-x0[2])/dq), ((x2[3]-x0[3])/dq),((x2[4]-x0[4])/dq), ((x2[5]-x0[5])/dq),
             ((x2[6]-x0[6])/dq), ((x2[7]-x0[7])/dq),((x2[8]-x0[8])/dq), ((x2[9]-x0[9])/dq)])#derivative with the second parameter
v3=np.array([((x3[0]-x0[0])/dq), ((x3[1]-x0[1])/dq),((x3[2]-x0[2])/dq), ((x3[3]-x0[3])/dq),((x3[4]-x0[4])/dq),
             ((x3[5]-x0[5])/dq),((x3[6]-x0[6])/dq), ((x3[7]-x0[7])/dq),((x3[8]-x0[8])/dq),((x3[9]-x0[9])/dq)])#derivative with the third parameter

v4= np.array([((x4[0]-x0[0])/dq), ((x4[1]-x0[1])/dq),((x4[2]-x0[2])/dq), ((x4[3]-x0[3])/dq),((x4[4]-x0[4])/dq), ((x4[5]-x0[5])/dq),((x4[6]-x0[6])/dq),
              ((x4[7]-x0[7])/dq),((x4[8]-x0[8])/dq), ((x4[9]-x0[9])/dq)])#derivative with the 4 parameter
v5=np.array([((x5[0]-x0[0])/dq), ((x5[1]-x0[1])/dq),((x5[2]-x0[2])/dq), ((x5[3]-x0[3])/dq),((x5[4]-x0[4])/dq),
             ((x5[5]-x0[5])/dq),((x5[6]-x0[6])/dq), ((x5[7]-x0[7])/dq),((x5[8]-x0[8])/dq),((x5[9]-x0[9])/dq)])#derivative with the third parameter

v6= np.array([((x6[0]-x0[0])/dq), ((x6[1]-x0[1])/dq),((x6[2]-x0[2])/dq), ((x6[3]-x0[3])/dq),((x6[4]-x0[4])/dq), ((x6[5]-x0[5])/dq),((x6[6]-x0[6])/dq),
              ((x6[7]-x0[7])/dq),((x6[8]-x0[8])/dq), ((x6[9]-x0[9])/dq)])#deriv

derv1 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v1[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v1[5:]))#derivative of function with the first parameter
derv2 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v2[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v2[5:]))#derivative of function with the second parameter
derv3 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v3[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v3[5:]))#derivative of function with the third parameter
derv4 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v4[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v4[5:]))#derivative of function with the 4 parameter
derv5 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v5[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v5[5:]))#derivative of function with the third parameter
derv6 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v6[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v6[5:]))#d
gradv=np.array([derv1,derv2,derv3,derv4,derv5,derv6])
norma=ln.norm(gradv)
print('norm of gradient in 0 step:')
print(norma)

thlist=([t[0],t[r1],t[r2],t[r3],t[r4],t[r5]])
fig1=[plt.plot(thlist,ulist1,"ro", lw=0.5),plt.plot(thlist,ulist2,"bo",lw=0.5)]
slist=[s0]
ilist=[i]
while(norma>=eps1):
    #s0list.append(s0)
    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD11')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u11=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])
     
    
    ivp=detest('2OD12')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem   
    
    u12=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])
    
   
    ivp=detest('2OD13')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    
    u13=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])        
  
    ivp=detest('2OD14')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    
    u14=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])


    ivp=detest('2OD22')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
   
    u22=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])
    
    ivp=detest('2OD23')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    
    u23=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

    ivp=detest('2OD24')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem     
    u24=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

    ivp=detest('2OD33')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem  
    u33=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

    ivp=detest('2OD34')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u34=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])
    
    ivp=detest('2OD44')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem      
    u44=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])    

    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD51')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u51=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD52')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u52=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD53')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u53=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])


    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD54')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u54=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD55')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u55=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])


    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD61')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u61=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD62')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u62=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD63')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u63=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD64')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u64=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD65')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem
    u65=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])


    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD66')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem    
    u66=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

    x11=np.array([u11[0][0],u11[1][0],u11[2][0],u11[3][0],u11[4][0], u11[0][1],u11[1][1],u11[2][1],u11[3][1],u11[4][1]])#solutions with with the first and second parameter changed by an amount dq 
    
    x12=np.array([u12[0][0],u12[1][0],u12[2][0],u12[3][0],u12[4][0],u12[0][1],u12[1][1],u12[2][1],u12[3][1],u12[4][1]])#solutions with with the first and third parameter changed by an amount dq 
    x13=np.array([u13[0][0],u13[1][0],u13[2][0],u13[3][0],u13[4][0], u13[0][1],u13[1][1],u13[2][1],u13[3][1],u13[4][1]])#solutions with with the second parameter changed by an amount 2dq
    
    x14=np.array([u14[0][0],u14[1][0],u14[2][0],u14[3][0],u14[4][0], u14[0][1],u14[1][1],u14[2][1],u14[3][1],u14[4][1]])#solutions with with the second and third parameter changed by an amount dq
    x22=np.array([u22[0][0],u22[1][0],u22[2][0],u22[3][0],u22[4][0], u22[0][1],u22[1][1],u22[2][1],u22[3][1],u22[4][1]])#solutions with with the third parameter changed by an amount 2dq
    
    x23=np.array([u23[0][0],u23[1][0],u23[2][0],u23[3][0],u23[4][0], u23[0][1],u23[1][1],u23[2][1],u23[3][1],u23[4][1]])#solutions with with the first parameter changed by an amount 2dq

    x24=np.array([u24[0][0],u24[1][0],u24[2][0],u24[3][0],u24[4][0],u24[0][1],u24[1][1],u24[2][1],u24[3][1],u24[4][1]])#solutions with with the first and second parameter changed by an amount dq 
    
    x33=np.array([u33[0][0],u33[1][0],u33[2][0],u33[3][0],u33[4][0],u33[0][1],u33[1][1],u33[2][1],u33[3][1],u33[4][1]])#solutions with with the first and third parameter changed by an amount dq 
    x34=np.array([u34[0][0],u34[1][0],u34[2][0],u34[3][0],u34[4][0], u34[0][1],u34[1][1],u34[2][1],u34[3][1],u34[4][1]])#

    x44=np.array([u44[0][0],u44[1][0],u44[2][0],u44[3][0],u44[4][0],u44[0][1],u44[1][1],u44[2][1],u44[3][1],u44[4][1]])#solutions with with the second parameter changed by an amount 2dq

    x51=np.array([u51[0][0],u51[1][0],u51[2][0],u51[3][0],u51[4][0], u51[0][1],u51[1][1],u51[2][1],u51[3][1],u51[4][1]])

    x52=np.array([u52[0][0],u52[1][0],u52[2][0],u52[3][0],u52[4][0],u52[0][1],u52[1][1],u52[2][1],u52[3][1],u52[4][1]])#solutions with with the first and third parameter changed by an amount dq 
    x53=np.array([u53[0][0],u53[1][0],u53[2][0],u53[3][0],u53[4][0], u53[0][1],u53[1][1],u53[2][1],u53[3][1],u53[4][1]])#solutions with with the second parameter changed by an amount 2dq
    
    x54=np.array([u54[0][0],u54[1][0],u54[2][0],u54[3][0],u54[4][0], u54[0][1],u54[1][1],u54[2][1],u54[3][1],u54[4][1]])#solutions with with the second and third parameter changed by an amount dq
    x55=np.array([u55[0][0],u55[1][0],u55[2][0],u55[3][0],u55[4][0], u55[0][1],u55[1][1],u55[2][1],u55[3][1],u55[4][1]])

    x61=np.array([u61[0][0],u61[1][0],u61[2][0],u61[3][0],u61[4][0], u61[0][1],u61[1][1],u61[2][1],u61[3][1],u61[4][1]])

    x62=np.array([u62[0][0],u62[1][0],u62[2][0],u62[3][0],u62[4][0],u62[0][1],u62[1][1],u62[2][1],u62[3][1],u62[4][1]])#solutions with with the first and third parameter changed by an amount dq 
    x63=np.array([u63[0][0],u63[1][0],u63[2][0],u63[3][0],u63[4][0], u63[0][1],u63[1][1],u63[2][1],u63[3][1],u63[4][1]])#solutions with with the second parameter changed by an amount 2dq
    
    x64=np.array([u64[0][0],u64[1][0],u64[2][0],u64[3][0],u64[4][0], u64[0][1],u64[1][1],u64[2][1],u64[3][1],u64[4][1]])#solutions with with the second and third parameter changed by an amount dq
    x65=np.array([u65[0][0],u65[1][0],u65[2][0],u65[3][0],u65[4][0], u65[0][1],u65[1][1],u65[2][1],u65[3][1],u65[4][1]])

    x66=np.array([u66[0][0],u66[1][0],u66[2][0],u66[3][0],u66[4][0], u66[0][1],u66[1][1],u66[2][1],u66[3][1],u66[4][1]])

    v11=np.array([((x11[0]-(2*x1[0]-x0[0]))/((dq)**2)), ((x11[1]-(2*x1[1]-x0[1]))/((dq)**2)),((x11[2]-(2*x1[2]-x0[2]))/((dq)**2)),
                           ((x11[3]-(2*x1[3]-x0[3]))/((dq)**2)),((x11[4]-(2*x1[4]-x0[4]))/((dq)**2)),((x11[5]-(2*x1[5]-x0[5]))/((dq)**2)),
                           ((x11[6]-(2*x1[6]-x0[6]))/((dq)**2)),((x11[7]-(2*x1[7]-x0[7]))/((dq)**2)), ((x11[8]-(2*x1[8]-x0[8]))/((dq)**2)),
                  ((x11[9]-(2*x1[9]-x0[9]))/((dq)**2))])#second derivative with 1 parameter of function
         
    v12=np.array([((x12[0]-x1[0]-x2[0]+x0[0])/((dq)**2)), ((x12[1]-x1[1]-x2[1]+x0[1])/((dq)**2)),((x12[2]-x1[2]-x2[2]+x0[2])/((dq)**2)),
                           ((x12[3]-x1[3]-x2[3]+x0[3])/((dq)**2)),((x12[4]-x1[4]-x2[4]+x0[4])/((dq)**2)), ((x12[5]-x1[5]-x2[5]+x0[5])/((dq)**2)),
                               ((x12[6]-x1[6]-x2[6]+x0[6])/((dq)**2)),((x12[7]-x1[7]-x2[7]+x0[7])/((dq)**2)),
                  ((x12[8]-x1[8]-x2[8]+x0[8])/((dq)**2)),((x12[9]-x1[9]-x2[9]+x0[9])/((dq)**2))])#derivative with 1 and 2 parameter of function 
               
    v13=np.array([((x13[0]-x1[0]-x3[0]+x0[0])/((dq)**2)), ((x13[1]-x1[1]-x3[1]+x0[1])/((dq)**2)),((x13[2]-x1[2]-x3[2]+x0[2])/((dq)**2)),
                           ((x13[3]-x1[3]-x3[3]+x0[3])/((dq)**2)),((x13[4]-x1[4]-x3[4]+x0[4])/((dq)**2)), ((x13[5]-x1[5]-x3[5]+x0[5])/((dq)**2)),
            ((x13[6]-x1[6]-x3[6]+x0[6])/((dq)**2)),((x13[7]-x1[7]-x3[7]+x0[7])/((dq)**2)),
                  ((x13[8]-x1[8]-x3[8]+x0[8])/((dq)**2)),((x13[9]-x1[9]-x3[9]+x0[9])/((dq)**2))])#derivative with 1 and 3 parameter of function
           
    v22=np.array([((x22[0]-(2*x2[0]-x0[0]))/((dq)**2)), ((x22[1]-(2*x2[1]-x0[1]))/((dq)**2)),((x22[2]-(2*x2[2]-x0[2]))/((dq)**2)),
                           ((x22[3]-(2*x2[3]-x0[3]))/((dq)**2)),((x22[4]-(2*x2[4]-x0[4]))/((dq)**2)), ((x22[5]-(2*x2[5]-x0[5]))/((dq)**2)),
                 ((x22[6]-(2*x2[6]-x0[6]))/((dq)**2)),((x22[7]-(2*x2[7]-x0[7]))/((dq)**2)),
                  ((x22[8]-(2*x2[8]-x0[8]))/((dq)**2)),((x22[9]-(2*x2[9]-x0[9]))/((dq)**2))])#second derivative with 2 parameter of function
           
    v23=np.array([((x23[0]-x3[0]-x2[0]+x0[0])/((dq)**2)), ((x23[1]-x3[1]-x2[1]+x0[1])/((dq)**2)),((x23[2]-x3[2]-x2[2]+x0[2])/((dq)**2)),
                           ((x23[3]-x3[3]-x2[3]+x0[3])/((dq)**2)),((x23[4]-x3[4]-x2[4]+x0[4])/((dq)**2)), ((x23[5]-x3[5]-x2[5]+x0[5])/((dq)**2)),
                 ((x23[6]-x2[6]-x3[6]+x0[6])/((dq)**2)),((x23[7]-x2[7]-x3[7]+x0[7])/((dq)**2)), ((x23[8]-x2[8]-x3[8]+x0[8])/((dq)**2)),
                  ((x23[9]-x2[9]-x3[9]+x0[9])/((dq)**2))])#derivative with 2 and 3 parameter of function
         
    v33=np.array([((x33[0]-(2*x3[0]-x0[0]))/((dq)**2)), ((x33[1]-(2*x3[1]-x0[1]))/((dq)**2)),((x33[2]-(2*x3[2]-x0[2]))/((dq)**2)),
                              ((x33[3]-(2*x3[3]-x0[3]))/((dq)**2)),((x33[4]-(2*x3[4]-x0[4]))/((dq)**2)), ((x33[5]-(2*x3[5]-x0[5]))/((dq)**2)),
                 ((x33[6]-(2*x3[6]-x0[6]))/((dq)**2)),((x33[7]-(2*x3[7]-x0[7]))/((dq)**2)), ((x33[8]-(2*x3[8]-x0[8]))/((dq)**2)),
                  ((x33[9]-(2*x3[9]-x0[9]))/((dq)**2))])#second derivative with 3 parameter of function
          

    v14=np.array([((x14[0]-x1[0]-x4[0]+x0[0])/((dq)**2)), ((x14[1]-x1[1]-x4[1]+x0[1])/((dq)**2)),((x14[2]-x1[2]-x4[2]+x0[2])/((dq)**2)),
                           ((x14[3]-x1[3]-x4[3]+x0[3])/((dq)**2)),
                  ((x14[4]-x1[4]-x4[4]+x0[4])/((dq)**2)), ((x14[5]-x1[5]-x4[5]+x0[5])/((dq)**2)),
                 ((x14[6]-x1[6]-x4[6]+x0[6])/((dq)**2)),((x14[7]-x1[7]-x4[7]+x0[7])/((dq)**2)),
                  ((x14[8]-x1[8]-x4[8]+x0[8])/((dq)**2)),((x14[9]-x1[9]-x4[9]+x0[9])/((dq)**2))])#derivative with 1 and 4 parameter of function
    v24=np.array([((x24[0]-x2[0]-x4[0]+x0[0])/((dq)**2)), ((x24[1]-x2[1]-x4[1]+x0[1])/((dq)**2)),((x24[2]-x2[2]-x4[2]+x0[2])/((dq)**2)),
                           ((x24[3]-x2[3]-x4[3]+x0[3])/((dq)**2)),((x24[4]-x2[4]-x4[4]+x0[4])/((dq)**2)),((x24[5]-x2[5]-x4[5]+x0[5])/((dq)**2)),
                 ((x24[6]-x4[6]-x2[6]+x0[6])/((dq)**2)),((x24[7]-x4[7]-x2[7]+x0[7])/((dq)**2)),
                  ((x24[8]-x4[8]-x2[8]+x0[8])/((dq)**2)),((x24[9]-x4[9]-x2[9]+x0[9])/((dq)**2))])#second derivative with 2 and 4 parameter of function     
    
           
    v34=np.array([((x34[0]-x3[0]-x4[0]+x0[0])/((dq)**2)), ((x34[1]-x3[1]-x4[1]+x0[1])/((dq)**2)),((x34[2]-x3[2]-x4[2]+x0[2])/((dq)**2)),
                           ((x34[3]-x3[3]-x4[3]+x0[3])/((dq)**2)),((x34[4]-x3[4]-x4[4]+x0[4])/((dq)**2)), ((x34[5]-x3[5]-x4[5]+x0[5])/((dq)**2)),
                  ((x34[6]-x3[6]-x4[6]+x0[6])/((dq)**2)),((x34[7]-x3[7]-x4[7]+x0[7])/((dq)**2)), ((x34[8]-x3[8]-x4[8]+x0[8])/((dq)**2)),
                  ((x34[9]-x3[9]-x4[9]+x0[9])/((dq)**2))])#derivative with 3 and 4 parameter of function
         
    v44=np.array([((x44[0]-(2*x4[0]-x0[0]))/((dq)**2)), ((x44[1]-(2*x4[1]-x0[1]))/((dq)**2)),((x44[2]-(2*x4[2]-x0[2]))/((dq)**2)),
                              ((x44[3]-(2*x4[3]-x0[3]))/((dq)**2)),((x44[4]-(2*x4[4]-x0[4]))/((dq)**2)), ((x44[5]-(2*x4[5]-x0[5]))/((dq)**2)),
                 ((x44[6]-(2*x4[6]-x0[6]))/((dq)**2)),((x44[7]-(2*x4[7]-x0[7]))/((dq)**2)), ((x44[8]-(2*x4[8]-x0[8]))/((dq)**2)),
                  ((x44[9]-(2*x4[9]-x0[9]))/((dq)**2))])#second derivative with 4 parameter of function



    v51=np.array([((x51[0]-x1[0]-x5[0]+x0[0])/((dq)**2)), ((x51[1]-x1[1]-x5[1]+x0[1])/((dq)**2)),((x51[2]-x1[2]-x5[2]+x0[2])/((dq)**2)),
                           ((x51[3]-x1[3]-x5[3]+x0[3])/((dq)**2)),((x51[4]-x1[4]-x5[4]+x0[4])/((dq)**2)), ((x51[5]-x1[5]-x5[5]+x0[5])/((dq)**2)),
                               ((x51[6]-x1[6]-x5[6]+x0[6])/((dq)**2)),((x51[7]-x1[7]-x5[7]+x0[7])/((dq)**2)),
                  ((x51[8]-x1[8]-x5[8]+x0[8])/((dq)**2)),((x51[9]-x1[9]-x5[9]+x0[9])/((dq)**2))])

    v52=np.array([((x52[0]-x2[0]-x5[0]+x0[0])/((dq)**2)), ((x52[1]-x2[1]-x5[1]+x0[1])/((dq)**2)),((x52[2]-x2[2]-x5[2]+x0[2])/((dq)**2)),
                           ((x52[3]-x2[3]-x5[3]+x0[3])/((dq)**2)),((x52[4]-x2[4]-x5[4]+x0[4])/((dq)**2)), ((x52[5]-x2[5]-x5[5]+x0[5])/((dq)**2)),
                               ((x52[6]-x2[6]-x5[6]+x0[6])/((dq)**2)),((x52[7]-x2[7]-x5[7]+x0[7])/((dq)**2)),
                  ((x52[8]-x2[8]-x5[8]+x0[8])/((dq)**2)),((x52[9]-x2[9]-x5[9]+x0[9])/((dq)**2))])


    v53=np.array([((x53[0]-x3[0]-x5[0]+x0[0])/((dq)**2)), ((x53[1]-x3[1]-x5[1]+x0[1])/((dq)**2)),((x53[2]-x3[2]-x5[2]+x0[2])/((dq)**2)),
                           ((x53[3]-x3[3]-x5[3]+x0[3])/((dq)**2)),((x53[4]-x3[4]-x5[4]+x0[4])/((dq)**2)), ((x53[5]-x3[5]-x5[5]+x0[5])/((dq)**2)),
                               ((x53[6]-x3[6]-x5[6]+x0[6])/((dq)**2)),((x53[7]-x3[7]-x5[7]+x0[7])/((dq)**2)),
                  ((x53[8]-x3[8]-x5[8]+x0[8])/((dq)**2)),((x53[9]-x3[9]-x5[9]+x0[9])/((dq)**2))])

    v54=np.array([((x54[0]-x4[0]-x5[0]+x0[0])/((dq)**2)), ((x54[1]-x4[1]-x5[1]+x0[1])/((dq)**2)),((x54[2]-x4[2]-x5[2]+x0[2])/((dq)**2)),
                           ((x54[3]-x4[3]-x5[3]+x0[3])/((dq)**2)),((x54[4]-x4[4]-x5[4]+x0[4])/((dq)**2)), ((x54[5]-x4[5]-x5[5]+x0[5])/((dq)**2)),
                               ((x54[6]-x4[6]-x5[6]+x0[6])/((dq)**2)),((x54[7]-x4[7]-x5[7]+x0[7])/((dq)**2)),
                  ((x54[8]-x4[8]-x5[8]+x0[8])/((dq)**2)),((x54[9]-x4[9]-x5[9]+x0[9])/((dq)**2))])

    v55=np.array([((x55[0]-(2*x5[0]-x0[0]))/((dq)**2)), ((x55[1]-(2*x5[1]-x0[1]))/((dq)**2)),((x55[2]-(2*x5[2]-x0[2]))/((dq)**2)),
                              ((x55[3]-(2*x5[3]-x0[3]))/((dq)**2)),((x55[4]-(2*x5[4]-x0[4]))/((dq)**2)), ((x55[5]-(2*x5[5]-x0[5]))/((dq)**2)),
                 ((x55[6]-(2*x5[6]-x0[6]))/((dq)**2)),((x55[7]-(2*x5[7]-x0[7]))/((dq)**2)), ((x55[8]-(2*x5[8]-x0[8]))/((dq)**2)),
                  ((x55[9]-(2*x5[9]-x0[9]))/((dq)**2))])#second derivative with 4 parameter of function



    v61=np.array([((x61[0]-x1[0]-x6[0]+x0[0])/((dq)**2)), ((x61[1]-x1[1]-x6[1]+x0[1])/((dq)**2)),((x61[2]-x1[2]-x6[2]+x0[2])/((dq)**2)),
                           ((x61[3]-x1[3]-x6[3]+x0[3])/((dq)**2)),((x61[4]-x1[4]-x6[4]+x0[4])/((dq)**2)), ((x61[5]-x1[5]-x6[5]+x0[5])/((dq)**2)),
                               ((x61[6]-x1[6]-x6[6]+x0[6])/((dq)**2)),((x61[7]-x1[7]-x6[7]+x0[7])/((dq)**2)),
                  ((x61[8]-x1[8]-x6[8]+x0[8])/((dq)**2)),((x61[9]-x1[9]-x6[9]+x0[9])/((dq)**2))])

    v62=np.array([((x62[0]-x2[0]-x6[0]+x0[0])/((dq)**2)), ((x62[1]-x2[1]-x6[1]+x0[1])/((dq)**2)),((x62[2]-x2[2]-x6[2]+x0[2])/((dq)**2)),
                           ((x62[3]-x2[3]-x6[3]+x0[3])/((dq)**2)),((x62[4]-x2[4]-x6[4]+x0[4])/((dq)**2)), ((x62[5]-x2[5]-x6[5]+x0[5])/((dq)**2)),
                               ((x62[6]-x2[6]-x6[6]+x0[6])/((dq)**2)),((x62[7]-x2[7]-x6[7]+x0[7])/((dq)**2)),
                  ((x62[8]-x2[8]-x6[8]+x0[8])/((dq)**2)),((x62[9]-x2[9]-x6[9]+x0[9])/((dq)**2))])


    v63=np.array([((x63[0]-x3[0]-x6[0]+x0[0])/((dq)**2)), ((x63[1]-x3[1]-x6[1]+x0[1])/((dq)**2)),((x63[2]-x3[2]-x6[2]+x0[2])/((dq)**2)),
                           ((x63[3]-x3[3]-x6[3]+x0[3])/((dq)**2)),((x63[4]-x3[4]-x6[4]+x0[4])/((dq)**2)), ((x63[5]-x3[5]-x6[5]+x0[5])/((dq)**2)),
                               ((x63[6]-x3[6]-x6[6]+x0[6])/((dq)**2)),((x63[7]-x3[7]-x6[7]+x0[7])/((dq)**2)),
                  ((x63[8]-x3[8]-x6[8]+x0[8])/((dq)**2)),((x63[9]-x3[9]-x6[9]+x0[9])/((dq)**2))])

    v64=np.array([((x64[0]-x4[0]-x6[0]+x0[0])/((dq)**2)), ((x64[1]-x4[1]-x6[1]+x0[1])/((dq)**2)),((x64[2]-x4[2]-x6[2]+x0[2])/((dq)**2)),
                           ((x64[3]-x4[3]-x6[3]+x0[3])/((dq)**2)),((x64[4]-x4[4]-x6[4]+x0[4])/((dq)**2)), ((x64[5]-x4[5]-x6[5]+x0[5])/((dq)**2)),
                               ((x64[6]-x4[6]-x6[6]+x0[6])/((dq)**2)),((x64[7]-x4[7]-x6[7]+x0[7])/((dq)**2)),
                  ((x64[8]-x4[8]-x6[8]+x0[8])/((dq)**2)),((x64[9]-x4[9]-x6[9]+x0[9])/((dq)**2))])


    v65=np.array([((x65[0]-x6[0]-x5[0]+x0[0])/((dq)**2)), ((x65[1]-x6[1]-x5[1]+x0[1])/((dq)**2)),((x65[2]-x6[2]-x5[2]+x0[2])/((dq)**2)),
                           ((x65[3]-x6[3]-x5[3]+x0[3])/((dq)**2)),((x65[4]-x6[4]-x5[4]+x0[4])/((dq)**2)), ((x65[5]-x6[5]-x5[5]+x0[5])/((dq)**2)),
                               ((x65[6]-x6[6]-x5[6]+x0[6])/((dq)**2)),((x65[7]-x6[7]-x5[7]+x0[7])/((dq)**2)),
                  ((x65[8]-x6[8]-x5[8]+x0[8])/((dq)**2)),((x65[9]-x6[9]-x5[9]+x0[9])/((dq)**2))])
    v66=np.array([((x66[0]-(2*x6[0]-x0[0]))/((dq)**2)), ((x66[1]-(2*x6[1]-x0[1]))/((dq)**2)),((x66[2]-(2*x6[2]-x0[2]))/((dq)**2)),
                              ((x66[3]-(2*x6[3]-x0[3]))/((dq)**2)),((x66[4]-(2*x6[4]-x0[4]))/((dq)**2)), ((x66[5]-(2*x6[5]-x0[5]))/((dq)**2)),
                 ((x66[6]-(2*x6[6]-x0[6]))/((dq)**2)),((x66[7]-(2*x6[7]-x0[7]))/((dq)**2)), ((x66[8]-(2*x6[8]-x0[8]))/((dq)**2)),
                  ((x66[9]-(2*x6[9]-x0[9]))/((dq)**2))])#second derivative with 4 parameter of function
    derv11=delta1*sum(2*(v1[0:5])**2-2*(xfact[0:5]-x0[0:5])*v11[0:5])+delta2*sum(2*(v1[5:])**2-2*(xfact[5:]-x0[5:])*v11[5:])
    derv12=delta1*sum(2*(v1[0:5])*v2[0:5]-2*(xfact[0:5]-x0[0:5])*v12[0:5])+delta2*sum(2*(v1[5:])*v2[5:]-2*(xfact[5:]-x0[5:])*v12[5:])
    derv13=delta1*sum(2*(v1[0:5])*v3[0:5]-2*(xfact[0:5]-x0[0:5])*v13[0:5])+delta2*sum(2*(v1[5:])*v3[5:]-2*(xfact[5:]-x0[5:])*v13[5:])
    derv14=delta1*sum(2*(v1[0:5])*v4[0:5]-2*(xfact[0:5]-x0[0:5])*v14[0:5])+delta2*sum(2*(v1[5:])*v4[5:]-2*(xfact[5:]-x0[5:])*v14[5:])

    derv22=delta1*sum(2*(v2[0:5])**2-2*(xfact[0:5]-x0[0:5])*v22[0:5])+delta2*sum(2*(v2[5:])**2-2*(xfact[5:]-x0[5:])*v22[5:])
    derv23=delta1*sum(2*(v2[0:5])*v3[0:5]-2*(xfact[0:5]-x0[0:5])*v23[0:5])+delta2*sum(2*(v2[5:])*v3[5:]-2*(xfact[5:]-x0[5:])*v23[5:])
    derv24=delta1*sum(2*(v2[0:5])*v4[0:5]-2*(xfact[0:5]-x0[0:5])*v24[0:5])+delta2*sum(2*(v2[5:])*v4[5:]-2*(xfact[5:]-x0[5:])*v24[5:])

    derv33=delta1*sum(2*(v3[0:5])**2-2*(xfact[0:5]-x0[0:5])*v33[0:5])+delta2*sum(2*(v3[5:])**2-2*(xfact[5:]-x0[5:])*v33[5:])
    derv34=delta1*sum(2*(v3[0:5])*v4[0:5]-2*(xfact[0:5]-x0[0:5])*v34[0:5])+delta2*sum(2*(v3[5:])*v4[5:]-2*(xfact[5:]-x0[5:])*v34[5:])

    derv44=delta1*sum(2*(v4[0:5])**2-2*(xfact[0:5]-x0[0:5])*v44[0:5])+delta2*sum(2*(v4[5:])**2-2*(xfact[5:]-x0[5:])*v44[5:])


    derv51=delta1*sum(2*(v1[0:5])*v5[0:5]-2*(xfact[0:5]-x0[0:5])*v51[0:5])+delta2*sum(2*(v1[5:])*v5[5:]-2*(xfact[5:]-x0[5:])*v51[5:])
    derv52=delta1*sum(2*(v5[0:5])*v2[0:5]-2*(xfact[0:5]-x0[0:5])*v52[0:5])+delta2*sum(2*(v5[5:])*v2[5:]-2*(xfact[5:]-x0[5:])*v52[5:])
    derv53=delta1*sum(2*(v5[0:5])*v3[0:5]-2*(xfact[0:5]-x0[0:5])*v53[0:5])+delta2*sum(2*(v5[5:])*v3[5:]-2*(xfact[5:]-x0[5:])*v53[5:])
    derv54=delta1*sum(2*(v5[0:5])*v4[0:5]-2*(xfact[0:5]-x0[0:5])*v54[0:5])+delta2*sum(2*(v5[5:])*v4[5:]-2*(xfact[5:]-x0[5:])*v54[5:])
    derv55=delta1*sum(2*(v5[0:5])**2-2*(xfact[0:5]-x0[0:5])*v55[0:5])+delta2*sum(2*(v5[5:])**2-2*(xfact[5:]-x0[5:])*v55[5:])

    derv61=delta1*sum(2*(v1[0:5])*v6[0:5]-2*(xfact[0:5]-x0[0:5])*v61[0:5])+delta2*sum(2*(v1[5:])*v6[5:]-2*(xfact[5:]-x0[5:])*v61[5:])
    derv62=delta1*sum(2*(v6[0:5])*v2[0:5]-2*(xfact[0:5]-x0[0:5])*v62[0:5])+delta2*sum(2*(v6[5:])*v2[5:]-2*(xfact[5:]-x0[5:])*v62[5:])
    derv63=delta1*sum(2*(v6[0:5])*v3[0:5]-2*(xfact[0:5]-x0[0:5])*v63[0:5])+delta2*sum(2*(v6[5:])*v3[5:]-2*(xfact[5:]-x0[5:])*v63[5:])
    derv64=delta1*sum(2*(v6[0:5])*v4[0:5]-2*(xfact[0:5]-x0[0:5])*v64[0:5])+delta2*sum(2*(v6[5:])*v4[5:]-2*(xfact[5:]-x0[5:])*v64[5:])
    derv65=delta1*sum(2*(v6[0:5])*v5[0:5]-2*(xfact[0:5]-x0[0:5])*v65[0:5])+delta2*sum(2*(v6[5:])*v5[5:]-2*(xfact[5:]-x0[5:])*v65[5:])
    derv66=delta1*sum(2*(v6[0:5])**2-2*(xfact[0:5]-x0[0:5])*v66[0:5])+delta2*sum(2*(v6[5:])**2-2*(xfact[5:]-x0[5:])*v66[5:])

    hesse=np.array([[derv11,derv12,derv13,derv14,derv51,derv61],[derv12,derv22,derv23,derv24,derv52,derv62],[derv13,derv23,derv33,derv34,derv53,derv63],
                    [derv14,derv24,derv34,derv44,derv54,derv64],[derv51,derv52,derv53,derv54,derv55,derv65], [derv61,derv62,derv63,derv64,derv65,derv66] ])#The Matrix of Hesse
    
   
    E=np.diag((1.,1.,1.,1.,1.,1.))#identity matrix
    chhesse=hesse+mu0*E#modified Hessian matrix
    inversechhesse=np.linalg.inv(chhesse)# inverse matrix of modified Hessian matrix    
    p=np.dot(inversechhesse,gradv.T)#vector for calculating  new parameters   
    s1=s0.T-p#new parameters
    
    s0=s1.T
    rk44 = rk.loadRKM('RK44')#load runge-kutta, find solutions with new parameters
    ivp=detest('2OD')
    myivp=ivp#load initial parameters from problem
    t,u=rk44(myivp)   
    u6=np.array([u[r1],u[r2],u[r3],u[r4],u[r5]])#approximate soluton in moment of time 1,2,3
    
    #totalpogrlist2=[totalpogr2]
    thlist=([t[0],t[r1],t[r2],t[r3],t[r4],t[r5]]) 
    x0=np.array([u6[0][0],u6[1][0],u6[2][0],u6[3][0],u6[4][0],u6[0][1],u6[1][1],u6[2][1],u6[3][1],u6[4][1]])
    x01list1=([u0[0],u6[0][0],u6[1][0],u6[2][0],u6[3][0],u6[4][0]])#for plot of 1 region
    x01list2=([u0[1],u6[0][1],u6[1][1],u6[2][1],u6[3][1],u6[4][1]])#for plot of 2 region 
    
   

    pogr1=abs(x0[0]-xfact[0])#error in 1 time for 1 region
    pogr2=abs(x0[1]-xfact[1])#error in 2 time for 1 region
    pogr3=abs(x0[2]-xfact[2] )#error in 3 time for 1 region
    pogr4=abs(x0[3]-xfact[3])#error in 4 time for 1 region
    pogr5=abs(x0[4]-xfact[4] )#error in 5 time for 1 region
    
    pogr6=abs(x0[5]-xfact[5] )#error in 1 time for 2 region
    pogr7=abs(x0[6]-xfact[6] )#error in 2 time for 2 region
    pogr8=abs(x0[7]-xfact[7] )#error in 3 time for 2 region
    pogr9=abs(x0[8]-xfact[8] )#error in 4 time for 2 region
    pogr10=abs(x0[9]-xfact[9])#error in 5 time for 2 region



    vectpogr1=[pogr1,pogr2,pogr3,pogr4,pogr5]#vector of error for 1 region
    totalpogr1=ln.norm(vectpogr1)#total error for 1 region
   

    vectpogr2=[pogr6,pogr7,pogr8,pogr9,pogr10]#vector of error for 2 region
    totalpogr2=ln.norm(vectpogr2)#total error for 2 region



    func1=delta1*sum((xfact[0:5]-x0[0:5])**2)+delta2*sum((xfact[5:]-x0[5:])**2) #Find value of function with new parameters    
    funclist.append(func1)#value of a function with new parameters       
    i=i+1 #element number in the array of value of a functions with new parameters

    if (funclist[i-1]>funclist[i]):
                s0=s1.T
                #s0=s0 #take new values of the parameters
                mu=(mu0)/2#halve the step
                mu0=mu
                #Find gradien and norm of gradient and return to the beginning of the cycle to calculate the Hesse matrix      
                rk44 = rk.loadRKM('RK44')#load runge-kutta                
                
                ivp=detest('2OD1')
                myivp=ivp#load initial parameters from problem
                t,u = rk44(myivp)#approximate soluton of problem             
                u7=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])
               
                ivp=detest('2OD2')
                myivp=ivp#load initial parameters from problem
                t,u = rk44(myivp)#approximate soluton of problem
                u8=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])
                
                ivp=detest('2OD3')
                myivp=ivp#load initial parameters from problem
                t,u = rk44(myivp)#approximate soluton of problem
                
                u9=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])
                
                ivp=detest('2OD4')
                myivp=ivp#load initial parameters from problem
                t,u = rk44(myivp)#approximate soluton of problem
               
                u40=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])
                ivp=detest('2OD5')
                myivp=ivp#load initial parameters from problem
                t,u = rk44(myivp)#approximate soluton of problem
                u50=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])

                ivp=detest('2OD6')
                myivp=ivp#load initial parameters from problem
                t,u = rk44(myivp)#approximate soluton of problem
                u60=np.array([u[r1], u[r2], u[r3],u[r4], u[r5]])




                x1=np.array([u7[0][0],u7[1][0],u7[2][0],u7[3][0],u7[4][0],u7[0][1],u7[1][1],u7[2][1],u7[3][1],u7[4][1]])#solutions with with the first parameter changed by an amount dq 
                x2=np.array([u8[0][0],u8[1][0],u8[2][0],u8[3][0],u8[4][0],u8[0][1],u8[1][1],u8[2][1],u8[3][1],u8[4][1]])#solutions with with the second parameter changed by an amount dq
                x3=np.array([u9[0][0],u9[1][0],u9[2][0],u9[3][0],u9[4][0], u9[0][1],u9[1][1],u9[2][1],u9[3][1],u9[4][1]])#solutions with with the third parameter changed by an amount dq
                x4=np.array([u40[0][0],u40[1][0],u40[2][0],u40[3][0],u40[4][0],u40[0][1],u40[1][1],u40[2][1],u40[3][1],u40[4][1]])#solutions with with the 4 parameter changed by an amount dq
                x5=np.array([u50[0][0],u50[1][0],u50[2][0],u50[3][0],u50[4][0],u50[0][1],u50[1][1],u50[2][1],u50[3][1],u50[4][1]])#solutions with with the third parameter changed by an amount dq
                x6=np.array([u60[0][0],u60[1][0],u60[2][0],u60[3][0],u60[4][0], u60[0][1],u60[1][1],u60[2][1],u60[3][1],u60[4][1]])         
                print('value for changing the Hessian matrix in 0 step:')
                print(mu0)
                                  
                v1=np.array([((x1[0]-x0[0])/dq), ((x1[1]-x0[1])/dq),((x1[2]-x0[2])/dq), ((x1[3]-x0[3])/dq),((x1[4]-x0[4])/dq), ((x1[5]-x0[5])/dq),
                             ((x1[6]-x0[6])/dq), ((x1[7]-x0[7])/dq),((x1[8]-x0[8])/dq), ((x1[9]-x0[9])/dq) ])#derivative with the first parameter
                v2=np.array([((x2[0]-x0[0])/dq), ((x2[1]-x0[1])/dq),((x2[2]-x0[2])/dq), ((x2[3]-x0[3])/dq),((x2[4]-x0[4])/dq),
                             ((x2[5]-x0[5])/dq),((x2[6]-x0[6])/dq), ((x2[7]-x0[7])/dq),((x2[8]-x0[8])/dq), ((x2[9]-x0[9])/dq)])#derivative with the second parameter
                v3=np.array([((x3[0]-x0[0])/dq), ((x3[1]-x0[1])/dq),((x3[2]-x0[2])/dq), ((x3[3]-x0[3])/dq),((x3[4]-x0[4])/dq),
                             ((x3[5]-x0[5])/dq),((x3[6]-x0[6])/dq), ((x3[7]-x0[7])/dq),((x3[8]-x0[8])/dq), ((x3[9]-x0[9])/dq)])#derivative with the third parameter
                v4= np.array([((x4[0]-x0[0])/dq), ((x4[1]-x0[1])/dq),((x4[2]-x0[2])/dq), ((x4[3]-x0[3])/dq),((x4[4]-x0[4])/dq),
                              ((x4[5]-x0[5])/dq),((x4[6]-x0[6])/dq), ((x4[7]-x0[7])/dq),((x4[8]-x0[8])/dq), ((x4[9]-x0[9])/dq)])#derivative with the 4 parameter


                v5=np.array([((x5[0]-x0[0])/dq), ((x5[1]-x0[1])/dq),((x5[2]-x0[2])/dq), ((x5[3]-x0[3])/dq),((x5[4]-x0[4])/dq),
                              ((x5[5]-x0[5])/dq),((x5[6]-x0[6])/dq), ((x5[7]-x0[7])/dq),((x5[8]-x0[8])/dq),((x5[9]-x0[9])/dq)])#derivative with the third parameter

                v6= np.array([((x6[0]-x0[0])/dq), ((x6[1]-x0[1])/dq),((x6[2]-x0[2])/dq), ((x6[3]-x0[3])/dq),((x6[4]-x0[4])/dq), ((x6[5]-x0[5])/dq),((x6[6]-x0[6])/dq),
                               ((x6[7]-x0[7])/dq),((x6[8]-x0[8])/dq), ((x6[9]-x0[9])/dq)])#deriv
                derv1 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v1[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v1[5:]))#derivative of function with the first parameter
                derv2 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v2[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v2[5:]))#derivative of function with the second parameter
                derv3 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v3[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v3[5:]))#derivative of function with the third parameter                
                derv4 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v4[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v4[5:]))#derivative of function with the 4 parameter                 
                derv5 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v5[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v5[5:]))#derivative of function with the third parameter
                derv6 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v6[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v6[5:]))#d
                gradv=np.array([derv1,derv2,derv3,derv4,derv5,derv6])                               
                norma=ln.norm(gradv)
                
                slist.append(s0)
                
    if (funclist[i-1]<=funclist[i]):
              s0=s1.T+p.T#new parameters              
              mu=(mu0)*2#double the step
              mu0=mu #take the previous values of the parameters and return to the beginning of the cycle to recalculate the Hesse matrix 
              
         
    
      
    s00=s0[0]
    s01=s0[1]
    beta1=s0[2]
    beta2=s0[3]
    k1=s0[4]
    k2=s0[5]
    
    ilist.append(i)
    totalpogrlist1.append(totalpogr1)#for plot of total error for 1 region
    totalpogrlist2.append(totalpogr2)#for plot of total error for 2 region
    print('step of iteration')
    print(i)
    print('value for changing the Hessian matrix or step:')
    print(mu0)
    print('value of function which must be minimized:')
    print(func1)
    print('norm of gradient:')
    print(norma)
    print('results of parameters identification:')
    print(s00,s01,beta1,beta2,k1,k2)
    print('fact value:') 
    print(xfact)
    print('solution:') 
    print(x1)
    print('vector of gradient:')
    print(gradv)
    print('total error 1:')
    print(totalpogr1)
    print('total error 2:')
    print(totalpogr2)            





maxpogr1=max(totalpogrlist1)#maximum total error
maxpogr2=max(totalpogrlist2)#maximum total error
maxpogr=max(maxpogr1,maxpogr2)



for s0 in slist:#plot for solutions with new parameters in i step
     
     
     ivp=detest('2OD')
     myivp1=ivp#load initial parameters from problem
     t,u = rk44(myivp1)#approximate soluton of problem  
     plt.axis([0,6,30,110])    
    
     fig1.append(plt.plot(thlist,x01list1,"go",lw=0.007))#for plot of 1 region
     fig1.append(plt.plot(thlist,x01list2,"yo",lw=0.007))#for plot of 2 region

             

plt.xlabel('t')
plt.ylabel('u')
plt.legend(['initial value 1', 'initial value 2','u1(t)', 'u2(t)' ], loc = 'upper right')              
plt.grid(fig1)
plt.show(fig1)

fig2=[]
plt.axis([0,max(ilist),0,maxpogr+1])
for i in ilist:#plots for total error for 1 and 2 regions with new parameters in i step
   plt.plot(i,totalpogrlist1[i],"ko",lw=0.07)#plot for total error for 1 regions with new parameters in i step
   plt.plot(i,totalpogrlist2[i],"ro",lw=0.07)#plot for total error for 2 regions with new parameters in i step
   
   fig2.append(plt.plot(i, totalpogrlist1[i],"ko",lw=0.007))
   fig2.append(plt.plot(i,totalpogrlist2[i],"ro", lw=0.007))



plt.xlabel('i')
plt.ylabel('totalpogr_i, i=1,2')
plt.legend(['total error1','total error2'], loc = 'upper right')              
plt.grid(fig2)
plt.show(fig2)
