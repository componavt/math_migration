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

u0=np.array([105.139101,37.694401])#initial value of population in 1 and 2 region respectively
u1= np.array([105.242686,37.623747])#numer of population, time 1
u2= np.array([105.386177,37.670206])#numer of population, time 2
u3= np.array([105.619818,37.727241])#numer of population, time 3
u4= np.array([105.913327,37.753604])#numer of population, time 4
u5= np.array([106.173357,37.799043])#numer of population, time 5 
ulist1=([u0[0], u1[0], u2[0],u3[0],u4[0],u5[0]])
ulist2=([u0[1], u1[1], u2[1],u3[1],u4[1],u5[1]])
v=np.array([0.72,0.79])# true utility of 1 region, 2 regions 
b1=0.013#average birth in 1 region
b2=0.014#average birth in 2 region

m1=0.013#average mortality 1 region
m2=0.013#average mortality 1 region

beta1=0.005 #parameter betta1    
beta2=0.002#parameter betta2 
dq=10**(-4)
q=1#positive parameter 
funclist=[ ]
t = np.linspace(0.,6.)
delta1=0.5#weight coefficient for function, which must be minimized,delta1+delta2=1
delta2=0.5#weight coefficient for function, which must be minimized,delta1+delta2=1 
s00= 0.008 #coefficient of migration  from 1 region to 2
s01=0.007  #coefficient of migration  from 2 region to 1
s0=np.array([s00,s01,beta1,beta2])#vector of initial parameters      
s0list=[s0]

class IVPOPTIM(object):  #Problems that are solved by the Runge-Kutta method
     def __init__(self, f=None,u0=np.array([105.139101,37.694401]) , t0=0., T=6.0, dt0=0.01, exact=None, desc='', name='',dt=0.01):
        self.u0  = u0
        self.rhs = f
        self.T   = T
        self.exact = exact
        self.description = desc
        self.t0 = t0
        self.dt0 = 0.01
        self.dt=0.01
        self.name = name
     def __repr__(self):
           return 'Problem Name:  '+self.name+'\n'+'Description:   '+self.description

def detest(testkey):#Description of problem
      import numpy as np
      ivpoptim=IVPOPTIM()
      if testkey=='2OD':#system of differential equations        
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=6.0#Final time of integration     

         ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                          exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+
                                                                                                       exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
         ivpoptim.dt0 = 0.01#time step
         ivpoptim.dt = 0.01

      elif testkey=='2OD1': # a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration

            ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1] +(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.01#time step
            ivpoptim.dt = 0.01
      elif testkey=='2OD2':# a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration 

            ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1] +s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.01#time step
            ivpoptim.dt = 0.01
      elif testkey=='2OD3':#system of differential equations        
             ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
             ivpoptim.T=6.0#Final time of integration     

             ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1] +s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1])]) 
             ivpoptim.dt0 = 0.01#time step
      elif testkey=='2OD4':#system of differential equations        
             ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
             ivpoptim.T=6.0#Final time of integration     

             ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1]+s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
             ivpoptim.dt0 = 0.01#time step
             ivpoptim.dt = 0.01
      elif testkey=='2OD11':# a system of differential equations with the first parameter changed by an amount dq
           ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
           ivpoptim.T=6.0#Final time of integration

           ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-(s0[0]+2*dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1]+(s0[0]+2*dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
           ivpoptim.dt0 = 0.01#time step
           ivpoptim.dt = 0.01
      elif testkey=='2OD12':# a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration

            ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1]+(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.1#time step
            
            ivpoptim.dt = 0.01
      elif testkey=='2OD13':# a system of differential equations with the first parameter changed by an amount dq
           ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
           ivpoptim.T=6.0#Final time of integration

           ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1]+(s0[0]+dq)*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1])]) 
           ivpoptim.dt0 = 0.1#time step
      elif testkey=='2OD14':# a system of differential equations with the first parameter changed by an amount dq
           ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
           ivpoptim.T=6.0#Final time of integration

           ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-(s0[0]+dq)*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1]+(s0[0]+dq)*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
           ivpoptim.dt0 = 0.01#time step
           ivpoptim.dt = 0.01

      elif testkey=='2OD22':# a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=6.0#Final time of integration

            ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+(s0[1]+2*dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]),
                                                 (b2*u[1]-m2*u[1] +s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-(s0[1]+2*dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
            ivpoptim.dt0 = 0.1#time step
            ivpoptim.dt = 0.01
      elif testkey=='2OD23':# a system of differential equations with the first parameter changed by an amount dq
          ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
          ivpoptim.T=6.0#Final time of integration

          ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]+(s0[1]+dq)*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1]+s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]-(s0[1]+dq)*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1])]) 
          ivpoptim.dt0 = 0.01#time step
          ivpoptim.dt = 0.01

      elif testkey=='2OD24':# a system of differential equations with the first parameter changed by an amount dq
          ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
          ivpoptim.T=6.0#Final time of integration

          ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1]+s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-(s0[1]+dq)*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
          ivpoptim.dt0 = 0.01#time step
          ivpoptim.dt = 0.01

      elif testkey=='2OD33':#system of differential equations        
          ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
          ivpoptim.T=6.0#Final time of integration     

          ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1] +s0[0]*((exp(q*v[1]*exp(-1/(s0[3]*u[1]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0]))))/(exp(q*v[1]*exp(-1/(s0[3]*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+2*dq)*u[0])))))*u[1])]) 
          ivpoptim.dt0 = 0.01#time step
          ivpoptim.dt = 0.01

      elif testkey=='2OD34':#system of differential equations        
          ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
          ivpoptim.T=6.0#Final time of integration     

          ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0] -s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1]+s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+dq)*u[1])))+exp(q*v[0]*exp(-1/((s0[2]+dq)*u[0])))))*u[1])]) 
          ivpoptim.dt0 = 0.01#time step   
          ivpoptim.dt = 0.01

      elif testkey=='2OD44':#system of differential equations        
          ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
          ivpoptim.T=6.0#Final time of integration     

          ivpoptim.rhs = lambda t,u: np.array([(b1*u[0]-m1*u[0]-s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]+s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1]), 
                                           (b2*u[1]-m2*u[1]+s0[0]*((exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1]))))/(exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[0]-s0[1]*((exp(q*v[0]*exp(-1/(s0[2]*u[0]))))/(exp(q*v[1]*exp(-1/((s0[3]+2*dq)*u[1])))+exp(q*v[0]*exp(-1/(s0[2]*u[0])))))*u[1])]) 
          ivpoptim.dt0 = 0.01#time step
          ivpoptim.dt = 0.01
      else: raise Exception('Unknown Detest problem')
      ivpoptim.name=testkey
      ivpoptim.description='Problem '+testkey+' of the non-stiff DETEST suite.'
      return ivpoptim


rk44 = rk.loadRKM('RK44')#load runge-kutta
ivp=detest('2OD')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
r=abs(t[1]-t[0])#the size of step
r1=int(round(1/r))
r2=2*r1
r3=3*r1
r4=4*r1
r5=5*r1
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

x0=np.array([u6[0][0],u6[1][0],u6[2][0],u6[3][0],u6[4][0],u6[0][1],u6[1][1],u6[2][1],u6[3][1],u6[4][1]])#array of solution in moment of time 1,2,3
xfact=np.array([u1[0],u2[0],u3[0],u4[0],u5[0],u1[1],u2[1],u3[1],u4[1],u5[1]])#fact values in moment of time 1,2,3
x1=np.array([u7[0][0],u7[1][0],u7[2][0],u7[3][0],u7[4][0],u7[0][1],u7[1][1],u7[2][1],u7[3][1],u7[4][1]])#solutions with with the first parameter changed by an amount dq 
x2=np.array([u8[0][0],u8[1][0],u8[2][0],u8[3][0],u8[4][0],u8[0][1],u8[1][1],u8[2][1],u8[3][1],u8[4][1]])#solutions with with the second parameter changed by an amount dq
x3=np.array([u9[0][0],u9[1][0],u9[2][0],u9[3][0],u9[4][0],u9[0][1],u9[1][1],u9[2][1],u9[3][1],u9[4][1]])#solutions with with the third parameter changed by an amount dq
x4=np.array([u40[0][0],u40[1][0],u40[2][0],u40[3][0],u40[4][0], u40[0][1],u40[1][1],u40[2][1],u40[3][1],u40[4][1]])#solutions with with the 4 parameter changed by an amount dq


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


derv1 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v1[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v1[5:]))#derivative of function with the first parameter
derv2 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v2[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v2[5:]))#derivative of function with the second parameter
derv3 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v3[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v3[5:]))#derivative of function with the third parameter
derv4 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v4[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v4[5:]))#derivative of function with the 4 parameter

gradv=np.array([derv1,derv2,derv3,derv4])
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
                  ((x34[6]-x3[6]-x4[3]+x0[6])/((dq)**2)),((x34[7]-x3[7]-x4[7]+x0[7])/((dq)**2)), ((x34[8]-x3[8]-x4[8]+x0[8])/((dq)**2)),
                  ((x34[9]-x3[9]-x4[9]+x0[9])/((dq)**2))])#derivative with 3 and 4 parameter of function
         
    v44=np.array([((x44[0]-(2*x4[0]-x0[0]))/((dq)**2)), ((x44[1]-(2*x4[1]-x0[1]))/((dq)**2)),((x44[2]-(2*x4[2]-x0[2]))/((dq)**2)),
                              ((x44[3]-(2*x4[3]-x0[3]))/((dq)**2)),((x44[4]-(2*x4[4]-x0[4]))/((dq)**2)), ((x44[5]-(2*x4[5]-x0[5]))/((dq)**2)),
                 ((x44[6]-(2*x4[6]-x0[6]))/((dq)**2)),((x44[7]-(2*x4[7]-x0[7]))/((dq)**2)), ((x44[8]-(2*x4[8]-x0[8]))/((dq)**2)),
                  ((x44[9]-(2*x4[9]-x0[9]))/((dq)**2))])#second derivative with 4 parameter of function


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

    hesse=np.array([[derv11,derv12,derv13,derv14],[derv12,derv22,derv23,derv24],[derv13,derv23,derv33,derv34],[derv14,derv24,derv34,derv44]])#The Matrix of Hesse
    
   
    E=np.diag((1.,1.,1.,1.))#identity matrix
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
                
                x1=np.array([u7[0][0],u7[1][0],u7[2][0],u7[3][0],u7[4][0],u7[0][1],u7[1][1],u7[2][1],u7[3][1],u7[4][1]])#solutions with with the first parameter changed by an amount dq 
                x2=np.array([u8[0][0],u8[1][0],u8[2][0],u8[3][0],u8[4][0],u8[0][1],u8[1][1],u8[2][1],u8[3][1],u8[4][1]])#solutions with with the second parameter changed by an amount dq
                x3=np.array([u9[0][0],u9[1][0],u9[2][0],u9[3][0],u9[4][0], u9[0][1],u9[1][1],u9[2][1],u9[3][1],u9[4][1]])#solutions with with the third parameter changed by an amount dq
                x4=np.array([u40[0][0],u40[1][0],u40[2][0],u40[3][0],u40[4][0],u40[0][1],u40[1][1],u40[2][1],u40[3][1],u40[4][1]])#solutions with with the 4 parameter changed by an amount dq
                         
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

                derv1 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v1[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v1[5:]))#derivative of function with the first parameter
                derv2 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v2[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v2[5:]))#derivative of function with the second parameter
                derv3 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v3[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v3[5:]))#derivative of function with the third parameter                
                derv4 = -2*delta1*sum((xfact[0:5]-x0[0:5])*(v4[0:5]))-2*delta2*sum((xfact[5:]-x0[5:])*(v4[5:]))#derivative of function with the 4 parameter                 
                gradv=np.array([derv1,derv2,derv3,derv4])                
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
    print(s00,s01,beta1,beta2)
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
