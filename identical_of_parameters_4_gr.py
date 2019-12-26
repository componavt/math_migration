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
eps1 =10#accuracy
eps2=0.1
a=10**(-8)#initial value of step for gradient
dr=0.1
u0=np.array([107.391,40.411])#initial value of population in 1 and 2 region respectively, 2004
u1= np.array([107.333,40.206])#numer of population, time, 2005
u2= np.array([107.998,39.892])#numer of population, time 2,2006
u3= np.array([106.668,39.636])#numer of population, time 3, 2007
u4= np.array([106.037,39.130])#numer of population, time 4,2008
u5= np.array([105.891,39.073])#numer of population, time 5,2009 




u6=np.array([105.317,38.851])#initial value of population in 1 and 2 region respectively, 2004
u7= np.array([104.808,38.666])#numer of population, time, 2005
u8= np.array([105.152,38.084])#numer of population, time 2,2006
u9= np.array([105.023,37.840])#numer of population, time 3, 2007
u10= np.array([105.002,37.746])#numer of population, time 4,2008
u11= np.array([105.040,37.697])#numer of population, time 5,2009 


u12=np.array([105.276,37.629])#numer of population, time 6,2010
u13= np.array([105.241,37.624])#numer of population, time 7,2011
u14= np.array([105.386,37.67])#numer of population, time 8,2012
u15= np.array([105.619,37.728])#numer of population, time 9,2013
u16= np.array([105.914,37.753])#numer of population, time 10,2014
u17= np.array([108.468,37.799])#numer of population, time 11,2015

ulist1=([u0[0], u1[0], u2[0],u3[0],u4[0],u5[0],u6[0], u7[0],
         u8[0],u9[0],u10[0],u11[0],u12[0], u13[0], u14[0],u15[0],u16[0],u17[0]])
ulist2=([u0[1], u1[1], u2[1],u3[1],u4[1],u5[1],u6[1], u7[1],
         u8[1],u9[1],u10[1],u11[1],u12[1], u13[1], u14[1],u15[1],u16[1],u17[1]])



k1=-0.0044  #average value of natural growth  in 1 region  
k2= -0.0019  #average value of natural growth  in 2 region

v=np.array([0.54, 0.55])#income ratio in the first and second region, respectively
 
c1=53.5#average number of economically active population in the first region

c2=20.1#average number of economically active population in the second region



beta1=0.79  #part of population  with income above the subsistence level in 1 region   
beta2=0.78#part of population  with income above the subsistence level in 2 region

#changing of 1-4 parameters:
dq1=4*10**(-4)
dq2=2*10**(-4)
dq3=4*10**(-4)
dq4=2*10**(-4)

totalpogrlist1=[]#initial element in array for figure of total error for 1 region
totalpogrlist2=[]#initial element in array for figure of total error for 2 region
funclist=[ ]
#t = np.linspace(0.,18.)
delta1=0.73 #weight coefficient for function, which must be minimized,delta1+delta2=1
delta2=0.27  #weight coefficient for function, which must be minimized,delta1+delta2=1 

s00=0.002 #coefficient of migration  from 1 region to 2
s01=0.009 #coefficient of migration  from 2 region to 1
print('введите значение для q из массива')
qlist=np.array([0.,0.1,1.,5.,10.,20.,30.])#psitive parameter in Bolzmann distribution
print(qlist)
q=float(input())
w0=0.0003443
w1=0.0003443
w2=0.000303
w3=0.00022838
w4=0.000452
w5=0.0009128
w6=0.0013484

if (q==qlist[0]):
     w=w0
if (q==qlist[1]): 
     w=w1
if (q==qlist[2]):
     w=w2
if (q==qlist[3]): 
     w=w3
if (q==qlist[4]):
     w=w4
if (q==qlist[5]): 
     w=w5
if (q==qlist[6]):
     w=w6

print(q, w) 
s0=np.array([s00,s01,k1,k2])#vector of initial parameters      
s0list=[s0]

class IVPOPTIM(object):  #Problems that are solved by the Runge-Kutta method
     def __init__(self, f=None,u0=np.array([107.391,40.411]) , t0=0., T=5000.0, dt0=0.001, exact=None, desc='', name='',dt=0.001):
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
         ivpoptim.T=5000.0#Final time of integration     

         ivpoptim.rhs = lambda t,u: np.array([(s0[2]*u[0]-s0[0]*((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[0] +
                                               s0[1]*((exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[1]), 
                                           (s0[3]*u[1]+s0[0]*((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[0]-
                                            s0[1]*((exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[1])])  





         ivpoptim.dt0 = 0.001#time step
         ivpoptim.dt = 0.001


      elif testkey=='2OD1': # a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=5000.0#Final time of integration
            ivpoptim.rhs = lambda t,u: np.array([(s0[2]*u[0]-(s0[0]+dq1)*((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[0] +
                                               s0[1]*((exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[1]), 
                                           (s0[3]*u[1]+(s0[0]+dq1)*((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[0]-
                                            s0[1]*((exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[1])])
            
            ivpoptim.dt0 = 0.001#time step
            ivpoptim.dt = 0.001
      elif testkey=='2OD2':# a system of differential equations with the first parameter changed by an amount dq
            ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
            ivpoptim.T=5000.0#Final time of integration 

            ivpoptim.rhs = lambda t,u: np.array([(s0[2]*u[0]-s0[0]*((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[0] +
                                               (s0[1]+dq2)*((exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[1]), 
                                           (s0[3]*u[1]+s0[0]*((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[0]-
                                            (s0[1]+dq2)*((exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[1])])
            ivpoptim.dt0 = 0.001#time step
            ivpoptim.dt = 0.001
      elif testkey=='2OD3':#system of differential equations        
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=5000.0#Final time of integration     

         ivpoptim.rhs = lambda t,u: np.array([((s0[2]+dq3)*u[0]-s0[0]*((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[0] +
                                               s0[1]*((exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[1]), 
                                           (s0[3]*u[1]+s0[0]*((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[0]-
                                            s0[1]*((exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[1])]) 

         ivpoptim.dt0 = 0.001#time step
         ivpoptim.dt = 0.001
      elif testkey=='2OD4':#system of differential equations        
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=5000.0#Final time of integration     

         ivpoptim.rhs = lambda t,u: np.array([(s0[2]*u[0]-s0[0]*((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[0] +
                                               s0[1]*((exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[1]), 
                                           ((s0[3]+dq4)*u[1]+s0[0]*((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[0]-
                                            s0[1]*((exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))/((exp(q*v[1]*(beta2*u[1]/(beta2*u[1]+c2))))+(exp(q*v[0]*(beta1*u[0]/(beta1*u[0]+c1))))))*u[1])])
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
r1=round(1/r)
r2=round(2/r)
r3=round(3/r)
r4=round(4/r)
r5=round(5/r)

r6=round(6/r)
r7=round(7/r)
r8=round(8/r)
r9=round(9/r)
r10=round(10/r)
r11=round(11/r)
                      
r12=round(12/r)
r13=round(13/r)
r14=round(14/r)
r15=round(15/r)
r16=round(16/r)
r17=round(17/r)                
                      

                      

u_solv=np.array([u[r1],u[r2],u[r3],u[r4],u[r5],u[r6],u[r7],u[r8],u[r9],u[r10], u[r11],u[r12],u[r13],u[r14],u[r15],u[r16],u[r17]])#approximate soluton in moment of time 1 - 17

x0=np.array([u_solv[0][0],u_solv[1][0],u_solv[2][0],u_solv[3][0],u_solv[4][0],u_solv[5][0],u_solv[6][0],u_solv[7][0],u_solv[8][0],u_solv[9][0],
                                u_solv[10][0],u_solv[11][0],u_solv[12][0],u_solv[13][0],u_solv[14][0],u_solv[15][0],u_solv[16][0],
                                u_solv[0][1],u_solv[1][1],u_solv[2][1],u_solv[3][1],u_solv[4][1],u_solv[5][1],u_solv[6][1],u_solv[7][1],u_solv[8][1],u_solv[9][1],
                                u_solv[10][1],u_solv[11][1],u_solv[12][1],u_solv[13][1],u_solv[14][1],u_solv[15][1],u_solv[16][1]])#array of approximate solution in moment of time 1-17



print(x0)
xfact=np.array([u1[0],u2[0],u3[0],u4[0],u5[0],u6[0],u7[0],u8[0],u9[0],u10[0],u11[0],
                u12[0],u13[0],u14[0],u15[0],u16[0],u17[0],
                u1[1],u2[1],u3[1],u4[1],u5[1],u6[1],u7[1],u8[1],u9[1],u10[1],u11[1],
                u12[1],u13[1],u14[1],u15[1],u16[1],u17[1]])#fact values in moment of time 1 - 17  

ivp=detest('2OD1')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
u_od_1=np.array([u[r1], u[r2], u[r3],u[r4], u[r5],u[r6],u[r7],u[r8],u[r9],u[r10], u[r11],
                u[r12],u[r13],u[r14],u[r15],u[r16],u[r17]])


ivp=detest('2OD2')
myivp=ivp  #load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
u_od_2=np.array([u[r1], u[r2], u[r3],u[r4], u[r5],u[r6],u[r7],u[r8],u[r9],u[r10], u[r11],u[r12],u[r13],u[r14],u[r15],u[r16],u[r17] ])

ivp=detest('2OD3')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
u_od_3=np.array([u[r1],u[r2],u[r3],u[r4],u[r5],u[r6],u[r7],u[r8],u[r9],u[r10], u[r11],u[r12],u[r13],u[r14],u[r15],u[r16],u[r17] ])

ivp=detest('2OD4')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
u_od_4=np.array([u[r1],u[r2],u[r3],u[r4],u[r5],u[r6],u[r7],u[r8],u[r9],u[r10], u[r11],u[r12],u[r13],u[r14],u[r15],u[r16],u[r17]])


#x1,x2,x3,x4 - solutions with change of 1,2,3,4 parameters with dq1,dq2,dq3,dq4 respectevely:

x1=np.array([u_od_1[0][0],u_od_1[1][0],u_od_1[2][0],u_od_1[3][0],u_od_1[4][0],u_od_1[5][0],u_od_1[6][0],u_od_1[7][0],u_od_1[8][0],u_od_1[9][0],
                                u_od_1[10][0],u_od_1[11][0],u_od_1[12][0],u_od_1[13][0],u_od_1[14][0],u_od_1[15][0],u_od_1[16][0],
                                u_od_1[0][1],u_od_1[1][1],u_od_1[2][1],u_od_1[3][1],u_od_1[4][1],u_od_1[5][1],u_od_1[6][1],u_od_1[7][1],u_od_1[8][1],u_od_1[9][1],
                                u_od_1[10][1],u_od_1[11][1],u_od_1[12][1],u_od_1[13][1],u_od_1[14][1],u_od_1[15][1],u_od_1[16][1]])


x2=np.array([u_od_2[0][0],u_od_2[1][0],u_od_2[2][0],u_od_2[3][0],u_od_2[4][0],u_od_2[5][0],u_od_2[6][0],u_od_2[7][0],u_od_2[8][0],u_od_2[9][0],
                                u_od_2[10][0],u_od_2[11][0],u_od_2[12][0],u_od_2[13][0],u_od_2[14][0],u_od_2[15][0],u_od_2[16][0],
                                u_od_2[0][1],u_od_2[1][1],u_od_2[2][1],u_od_2[3][1],u_od_2[4][1],u_od_2[5][1],u_od_2[6][1],u_od_2[7][1],u_od_2[8][1],u_od_2[9][1],
                                u_od_2[10][1],u_od_2[11][1],u_od_2[12][1],u_od_2[13][1],u_od_2[14][1],u_od_2[15][1],u_od_2[16][1]])





x3=np.array([u_od_3[0][0],u_od_3[1][0],u_od_3[2][0],u_od_3[3][0],u_od_3[4][0],u_od_3[5][0],u_od_3[6][0],u_od_3[7][0],u_od_3[8][0],u_od_3[9][0],
                                u_od_3[10][0],u_od_3[11][0],u_od_3[12][0],u_od_3[13][0],u_od_3[14][0],u_od_3[15][0],u_od_3[16][0],
                                u_od_3[0][1],u_od_3[1][1],u_od_3[2][1],u_od_3[3][1],u_od_3[4][1],u_od_3[5][1],u_od_3[6][1],u_od_3[7][1],u_od_3[8][1],u_od_3[9][1],
                                u_od_3[10][1],u_od_3[11][1],u_od_3[12][1],u_od_3[13][1],u_od_3[14][1],u_od_3[15][1],u_od_3[16][1]])
x4=np.array([u_od_4[0][0],u_od_4[1][0],u_od_4[2][0],u_od_4[3][0],u_od_4[4][0],u_od_4[5][0],u_od_4[6][0],u_od_4[7][0],u_od_4[8][0],u_od_4[9][0],
                                u_od_4[10][0],u_od_4[11][0],u_od_4[12][0],u_od_4[13][0],u_od_4[14][0],u_od_4[15][0],u_od_4[16][0],
                                u_od_4[0][1],u_od_4[1][1],u_od_4[2][1],u_od_4[3][1],u_od_4[4][1],u_od_4[5][1],u_od_4[6][1],u_od_4[7][1],u_od_4[8][1],u_od_4[9][1],
                                u_od_4[10][1],u_od_4[11][1],u_od_4[12][1],u_od_4[13][1],u_od_4[14][1],u_od_4[15][1],u_od_4[16][1]])


#arrays for figures of experimental data and model solution:  
x01list1=([u0[0],u_solv[0][0],u_solv[1][0],u_solv[2][0],u_solv[3][0],u_solv[4][0],u_solv[5][0],u_solv[6][0],u_solv[7][0],
                  u_solv[8][0],u_solv[9][0],u_solv[10][0],u_solv[11][0],u_solv[12][0],u_solv[13][0],u_solv[14][0],u_solv[15][0],u_solv[16][0]])
x01list2=([u0[1],u_solv[0][1],u_solv[1][1],u_solv[2][1],u_solv[3][1],u_solv[4][1],u_solv[5][1],u_solv[6][1],u_solv[7][1],
                 u_solv[8][1],u_solv[9][1],u_solv[10][1],u_solv[11][1],u_solv[12][1],u_solv[13][1],u_solv[14][1], u_solv[15][1],u_solv[16][1]])

i=0#number of iteration 
func=delta1*sum((xfact[0:17]-x0[0:17])**2)+delta2*sum((xfact[17:]-x0[17:])**2)#function which must be minimized

funclist=[func]#first element is value of a function with initial parameters

print('value of function which must be minimized in 0 step:')
print(func)
pogr1=abs(x0[0]-xfact[0])#error in 1 year  for 1 region
pogr2=abs(x0[1]-xfact[1])#error in 2 year for 1 region
pogr3=abs(x0[2]-xfact[2] )#error in 3 year for 1 region
pogr4=abs(x0[3]-xfact[3])#error in 4 year for 1 region
pogr5=abs(x0[4]-xfact[4])#error in 5 year for 1 region
pogr6=abs(x0[5]-xfact[5])#error in 6 year  for 1 region
pogr7=abs(x0[6]-xfact[6])#error in 7 year for 1 region
pogr8=abs(x0[7]-xfact[7] )#error in 8 year for 1 region
pogr9=abs(x0[8]-xfact[8])#error in 9 year for 1 region
pogr10=abs(x0[9]-xfact[9])#error in 10 year for 1 region
pogr11=abs(x0[10]-xfact[10])#error in 11 year for 1 region
pogr12=abs(x0[11]-xfact[11])#error in 12 year for 1 region
pogr13=abs(x0[12]-xfact[12])#error in 13 year  for 1 region
pogr14=abs(x0[13]-xfact[13])#error in 14 year for 1 region
pogr15=abs(x0[14]-xfact[14])#error in 15 year for 1 region
pogr16=abs(x0[15]-xfact[15])#error in 16 year  for 2 region
pogr17=abs(x0[16]-xfact[16])#error in 17 year for 2 region

pogr18=abs(x0[17]-xfact[17] )#error in 1 year for 2 region
pogr19=abs(x0[18]-xfact[18])#error in 2 year for 2 region
pogr20=abs(x0[19]-xfact[19])#error in 3 year for 2 region
pogr21=abs(x0[20]-xfact[20])#error in 4 year  for 2 region
pogr22=abs(x0[21]-xfact[21])#error in 5 year for 2 region
pogr23=abs(x0[22]-xfact[22] )#error in 6 year for 2 region
pogr24=abs(x0[23]-xfact[23])#error in 7 year for 2 region
pogr25=abs(x0[24]-xfact[24])#error in 8 year for 2 region
pogr26=abs(x0[25]-xfact[25])#error in 9 year for 2 region
pogr27=abs(x0[26]-xfact[26])#error in 10 year for 2 region
pogr28=abs(x0[27]-xfact[27])#error in 11 year  for 2 region
pogr29=abs(x0[28]-xfact[28])#error in 12 year for 2 region
pogr30=abs(x0[29]-xfact[29])#error in 13 year for 2 region
pogr31=abs(x0[30]-xfact[30])#error in 14 year for 2 region
pogr32=abs(x0[31]-xfact[31])#error in 15 year  for 2 region
pogr33=abs(x0[32]-xfact[32])#error in 16 year for 2 region
pogr34=abs(x0[33]-xfact[33])#error in 17 year for 2 region

vectpogr1=[pogr1,pogr2,pogr3,pogr4,pogr5,pogr6,pogr7,pogr8,pogr9,pogr10,pogr11,pogr12,pogr13,pogr14,pogr15,pogr16,pogr17]#vector of error for 1 region
totalpogr1=ln.norm(vectpogr1)#total error for 1 region
totalpogrlist1.append(totalpogr1)# for plot of total error for 1 region

vectpogr2=[pogr18,pogr19,pogr20,pogr21,pogr22,pogr23,pogr24,pogr25,pogr26,pogr27,pogr28,pogr29,pogr30,pogr31,pogr32,pogr33,pogr34]#vector of error for 2 region
totalpogr2=ln.norm(vectpogr2)#total error for 2 region
totalpogrlist2.append(totalpogr2)# for plot of total error for 2 region

#v1-v4 are arrays of derivatives solutions with changing parameters:
v1=np.array([((x1[0]-x0[0])/dq1), ((x1[1]-x0[1])/dq1),((x1[2]-x0[2])/dq1), ((x1[3]-x0[3])/dq1),
             ((x1[4]-x0[4])/dq1),  ((x1[5]-x0[5])/dq1),((x1[6]-x0[6])/dq1), ((x1[7]-x0[7])/dq1),
              ((x1[8]-x0[8])/dq1), ((x1[9]-x0[9])/dq1),((x1[10]-x0[10])/dq1), ((x1[11]-x0[11])/dq1),
              ((x1[12]-x0[12])/dq1), ((x1[13]-x0[13])/dq1),((x1[14]-x0[14])/dq1),((x1[15]-x0[15])/dq1),
               ((x1[16]-x0[16])/dq1), ((x1[17]-x0[17])/dq1),((x1[18]-x0[18])/dq1), ((x1[19]-x0[19])/dq1),             
               ((x1[20]-x0[20])/dq1), ((x1[21]-x0[21])/dq1),((x1[22]-x0[22])/dq1), ((x1[23]-x0[23])/dq1),
             ((x1[24]-x0[24])/dq1),  ((x1[25]-x0[25])/dq1),((x1[26]-x0[26])/dq1), ((x1[27]-x0[27])/dq1),
              ((x1[28]-x0[28])/dq1), ((x1[29]-x0[29])/dq1),((x1[30]-x0[30])/dq1), ((x1[31]-x0[31])/dq1),
             ((x1[32]-x0[32])/dq1), ((x1[33]-x0[33])/dq1)])#derivative with the first parameter
v2=np.array([((x2[0]-x0[0])/dq2), ((x2[1]-x0[1])/dq2),((x2[2]-x0[2])/dq4), ((x2[3]-x0[3])/dq4),
             ((x2[4]-x0[4])/dq2), ((x2[5]-x0[5])/dq2),((x2[6]-x0[6])/dq4), ((x2[7]-x0[7])/dq4),
               ((x2[8]-x0[8])/dq2), ((x2[9]-x0[9])/dq2),((x2[10]-x0[10])/dq2), ((x2[11]-x0[11])/dq2),
               ((x2[12]-x0[12])/dq2), ((x2[13]-x0[13])/dq2),((x2[14]-x0[14])/dq2),((x2[15]-x0[15])/dq2),
                ((x2[16]-x0[16])/dq2), ((x2[17]-x0[17])/dq2),((x2[18]-x0[18])/dq2), ((x2[19]-x0[19])/dq2),
               ((x2[20]-x0[20])/dq2), ((x2[21]-x0[21])/dq2),((x2[22]-x0[22])/dq2), ((x2[23]-x0[23])/dq2),
             ((x2[24]-x0[24])/dq2),  ((x2[25]-x0[25])/dq2),((x2[26]-x0[26])/dq2), ((x2[27]-x0[27])/dq2),
              ((x2[28]-x0[28])/dq2), ((x2[29]-x0[29])/dq2),((x2[30]-x0[30])/dq2), ((x2[31]-x0[31])/dq2),
             ((x2[32]-x0[32])/dq2), ((x2[33]-x0[33])/dq2)])#derivative with the second parameter


v3=np.array([((x3[0]-x0[0])/dq3), ((x3[1]-x0[1])/dq3),((x3[2]-x0[2])/dq3), ((x3[3]-x0[3])/dq3),
             ((x3[4]-x0[4])/dq3),  ((x3[5]-x0[5])/dq3),((x3[6]-x0[6])/dq3), ((x3[7]-x0[7])/dq3),
              ((x3[8]-x0[8])/dq3), ((x3[9]-x0[9])/dq3),((x3[10]-x0[10])/dq3), ((x3[11]-x0[11])/dq3),
              ((x3[12]-x0[12])/dq3), ((x3[13]-x0[13])/dq3),((x3[14]-x0[14])/dq3),((x3[15]-x0[15])/dq3),
               ((x3[16]-x0[16])/dq3), ((x3[17]-x0[17])/dq3),((x3[18]-x0[18])/dq3), ((x3[19]-x0[19])/dq3),             
               ((x3[20]-x0[20])/dq3), ((x3[21]-x0[21])/dq3),((x3[22]-x0[22])/dq3), ((x3[23]-x0[23])/dq3),
             ((x3[24]-x0[24])/dq3),  ((x3[25]-x0[25])/dq3),((x3[26]-x0[26])/dq3), ((x3[27]-x0[27])/dq3),
              ((x3[28]-x0[28])/dq3), ((x3[29]-x0[29])/dq3),((x3[30]-x0[30])/dq3), ((x3[31]-x0[31])/dq3),
              ((x3[32]-x0[32])/dq3), ((x3[33]-x0[33])/dq3)])#derivative with the third parameter
v4=np.array([((x4[0]-x0[0])/dq4), ((x4[1]-x0[1])/dq4),((x4[2]-x0[2])/dq2), ((x4[3]-x0[3])/dq2),
             ((x4[4]-x0[4])/dq4), ((x4[5]-x0[5])/dq4),((x4[6]-x0[6])/dq2), ((x4[7]-x0[7])/dq2),
               ((x4[8]-x0[8])/dq4), ((x4[9]-x0[9])/dq4),((x4[10]-x0[10])/dq2), ((x4[11]-x0[11])/dq4),
               ((x4[12]-x0[12])/dq4), ((x4[13]-x0[13])/dq4),((x4[14]-x0[14])/dq4),((x4[15]-x0[15])/dq4),
                ((x4[16]-x0[16])/dq4), ((x4[17]-x0[17])/dq4),((x4[18]-x0[18])/dq4), ((x4[19]-x0[19])/dq4),
               ((x4[20]-x0[20])/dq4), ((x4[21]-x0[21])/dq4),((x4[22]-x0[22])/dq4), ((x4[23]-x0[23])/dq4),
             ((x4[24]-x0[24])/dq4),  ((x4[25]-x0[25])/dq4),((x4[26]-x0[26])/dq4), ((x4[27]-x0[27])/dq4),
              ((x4[28]-x0[28])/dq4), ((x4[29]-x0[29])/dq4),((x4[30]-x0[30])/dq4), ((x4[31]-x0[31])/dq4),
              ((x4[32]-x0[32])/dq4), ((x4[33]-x0[33])/dq4)])#derivative with the 4-th parameter


# derv1 -derv4 are elements of gradient:
derv1 = -2*delta1*sum((xfact[0:17]-x0[0:17])*(v1[0:17]))-2*delta2*sum((xfact[17:]-x0[17:])*(v1[17:]))#derivative of function with the 1-th parameter
derv2 = -2*delta1*sum((xfact[0:17]-x0[0:17])*(v2[0:17]))-2*delta2*sum((xfact[17:]-x0[17:])*(v2[17:]))#derivative of function with the 2-th parameter
derv3 = -2*delta1*sum((xfact[0:17]-x0[0:17])*(v3[0:17]))-2*delta2*sum((xfact[17:]-x0[17:])*(v3[17:]))#derivative of function with the 3-th parameter
derv4 = -2*delta1*sum((xfact[0:17]-x0[0:17])*(v4[0:17]))-2*delta2*sum((xfact[17:]-x0[17:])*(v4[17:]))#derivative of function with the 4-th parameter



gradv=np.array([derv1,derv2,derv3,derv4])#gradient
norma=ln.norm(gradv)#norm of gradient
print('norm of gradient in 0 step:')
print(norma)
normlist=[norma]#first element of norma with initial parameters


thlist1=([t[r1],t[r2],t[r3],t[r4],t[r5],t[r6],t[r7],t[r8],t[r9],t[r10], t[r11],t[r12],t[r13],t[r14],t[r15],t[r16],t[r17]])     
thlist=([t[0],t[r1],t[r2],t[r3],t[r4],t[r5],t[r6],t[r7],t[r8],t[r9],t[r10], t[r11],t[r12],t[r13],t[r14],t[r15],t[r16],t[r17] ])
fig1=[plt.plot(thlist,ulist1,"ro", lw=0.5),plt.plot(thlist,ulist2,"bo",lw=0.5)]
slist=[s0]
ilist=[i]
eps1=normlist[0]*w            
     
while(norma>eps1): 
       i=i+1#number of iteration        
       ilist.append(i)
       s1=s0-a*gradv #set of new parameters     
       s0=s1       
       slist.append(s0)        
       
       
       s00=s0[0]
       s01=s0[1]
       k1=s0[2]
       k2=s0[3]
       
       s0=np.array([s00,s01,k1,k2])
       
       rk44 = rk.loadRKM('RK44')#load runge-kutta, find solutions with new parameters
       ivp=detest('2OD')
       myivp=ivp#load initial parameters from problem
       t,u=rk44(myivp)
       u_solv=np.array([u[r1],u[r2],u[r3],u[r4],u[r5],u[r6],u[r7],u[r8],u[r9],u[r10], u[r11],u[r12],u[r13],u[r14],u[r15],u[r16],u[r17] ])#approximate soluton in moment of time 1 - 11

       x0=np.array([u_solv[0][0],u_solv[1][0],u_solv[2][0],u_solv[3][0],u_solv[4][0],u_solv[5][0],u_solv[6][0],u_solv[7][0],u_solv[8][0],u_solv[9][0],
                                u_solv[10][0],u_solv[11][0],u_solv[12][0],u_solv[13][0],u_solv[14][0],u_solv[15][0],u_solv[16][0],
                                u_solv[0][1],u_solv[1][1],u_solv[2][1],u_solv[3][1],u_solv[4][1],u_solv[5][1],u_solv[6][1],u_solv[7][1],u_solv[8][1],u_solv[9][1],
                                u_solv[10][1],u_solv[11][1],u_solv[12][1],u_solv[13][1],u_solv[14][1],u_solv[15][1],u_solv[16][1]])    
    
       
       xfact=np.array([u1[0],u2[0],u3[0],u4[0],u5[0],u6[0],u7[0],u8[0],u9[0],u10[0],u11[0],
                u12[0],u13[0],u14[0],u15[0],u16[0],u17[0],
                u1[1],u2[1],u3[1],u4[1],u5[1],u6[1],u7[1],u8[1],u9[1],u10[1],u11[1],
                u12[1],u13[1],u14[1],u15[1],u16[1],u17[1] ])#fact values in moment of time 1 - 17
       
       func1=delta1*sum((xfact[0:17]-x0[0:17])**2)+delta2*sum((xfact[17:]-x0[17:])**2)#function which must be minimized
       print('func1')
       print(func1)
       funclist.append(func1)#value of a function with new parameters       
         
       
       thlist=([t[0],t[r1],t[r2],t[r3],t[r4],t[r5],t[r6],t[r7],t[r8],t[r9],t[r10], t[r11],t[r12],t[r13],t[r14],t[r15],t[r16],t[r17] ])
       x01list1=([u0[0],u_solv[0][0],u_solv[1][0],u_solv[2][0],u_solv[3][0],u_solv[4][0],u_solv[5][0],u_solv[6][0],u_solv[7][0],
                  u_solv[8][0],u_solv[9][0],u_solv[10][0],u_solv[11][0],u_solv[12][0],u_solv[13][0],u_solv[14][0],u_solv[15][0],u_solv[16][0]])
       x01list2=([u0[1],u_solv[0][1],u_solv[1][1],u_solv[2][1],u_solv[3][1],u_solv[4][1],u_solv[5][1],u_solv[6][1],u_solv[7][1],
                 u_solv[8][1],u_solv[9][1],u_solv[10][1],u_solv[11][1],u_solv[12][1],u_solv[13][1],u_solv[14][1], u_solv[15][1],u_solv[16][1]])
       
       
       pogr1=abs(x0[0]-xfact[0])#error in 1 year  for 1 region
       pogr2=abs(x0[1]-xfact[1])#error in 2 year for 1 region
       pogr3=abs(x0[2]-xfact[2] )#error in 3 year for 1 region
       pogr4=abs(x0[3]-xfact[3])#error in 4 year for 1 region
       pogr5=abs(x0[4]-xfact[4])#error in 5 year for 1 region
       pogr6=abs(x0[5]-xfact[5])#error in 6 year  for 1 region
       pogr7=abs(x0[6]-xfact[6])#error in 7 year for 1 region
       pogr8=abs(x0[7]-xfact[7] )#error in 8 year for 1 region
       pogr9=abs(x0[8]-xfact[8])#error in 9 year for 1 region
       pogr10=abs(x0[9]-xfact[9])#error in 10 year for 1 region
       pogr11=abs(x0[10]-xfact[10])#error in 11 year for 1 region
       pogr12=abs(x0[11]-xfact[11])#error in 12 year for 1 region
       pogr13=abs(x0[12]-xfact[12])#error in 13 year  for 1 region
       pogr14=abs(x0[13]-xfact[13])#error in 14 year for 1 region
       pogr15=abs(x0[14]-xfact[14])#error in 15 year for 1 region
       pogr16=abs(x0[15]-xfact[15])#error in 16 year  for 2 region
       pogr17=abs(x0[16]-xfact[16])#error in 17 year for 2 region

       pogr18=abs(x0[17]-xfact[17] )#error in 1 year for 2 region
       pogr19=abs(x0[18]-xfact[18])#error in 2 year for 2 region
       pogr20=abs(x0[19]-xfact[19])#error in 3 year for 2 region
       pogr21=abs(x0[20]-xfact[20])#error in 4 year  for 2 region
       pogr22=abs(x0[21]-xfact[21])#error in 5 year for 2 region
       pogr23=abs(x0[22]-xfact[22] )#error in 6 year for 2 region
       pogr24=abs(x0[23]-xfact[23])#error in 7 year for 2 region
       pogr25=abs(x0[24]-xfact[24])#error in 8 year for 2 region
       pogr26=abs(x0[25]-xfact[25])#error in 9 year for 2 region
       pogr27=abs(x0[26]-xfact[26])#error in 10 year for 2 region
       pogr28=abs(x0[27]-xfact[27])#error in 11 year  for 2 region
       pogr29=abs(x0[28]-xfact[28])#error in 12 year for 2 region
       pogr30=abs(x0[29]-xfact[29])#error in 13 year for 2 region
       pogr31=abs(x0[30]-xfact[30])#error in 14 year for 2 region
       pogr32=abs(x0[31]-xfact[31])#error in 15 year  for 2 region
       pogr33=abs(x0[32]-xfact[32])#error in 16 year for 2 region
       pogr34=abs(x0[33]-xfact[33])#error in 17 year for 2 region

       vectpogr1=[pogr1,pogr2,pogr3,pogr4,pogr5,pogr6,pogr7,pogr8,pogr9,pogr10,pogr11,pogr12,pogr13,pogr14,pogr15,pogr16,pogr17]#vector of error for 1 region
       totalpogr1=ln.norm(vectpogr1)#total error for 1 region
       totalpogrlist1.append(totalpogr1)# for plot of total error for 1 region

       vectpogr2=[pogr18,pogr19,pogr20,pogr21,pogr22,pogr23,pogr24,pogr25,pogr26,pogr27,pogr28,pogr29,pogr30,pogr31,pogr32,pogr33,pogr34]#vector of error for 2 region
       totalpogr2=ln.norm(vectpogr2)#total error for 2 region
       totalpogrlist2.append(totalpogr2)# for plot of total error for 2 region


       
       if (funclist[i]<funclist[i-1]):
      
                
                rk44 = rk.loadRKM('RK44')#load runge-kutta                
                ivp=detest('2OD1')
                myivp=ivp#load initial parameters from problem
                t,u = rk44(myivp)#approximate soluton of problem
                u_od_1=np.array([u[r1], u[r2], u[r3],u[r4], u[r5],u[r6],u[r7],u[r8],u[r9],u[r10], u[r11],
                                   u[r12],u[r13],u[r14],u[r15],u[r16],u[r17] ])


                ivp=detest('2OD2')
                myivp=ivp  #load initial parameters from problem
                t,u = rk44(myivp)#approximate soluton of problem
                u_od_2=np.array([u[r1], u[r2], u[r3],u[r4], u[r5],u[r6],u[r7],u[r8],u[r9],u[r10], u[r11],u[r12],u[r13],u[r14],u[r15],u[r16],u[r17] ])      

                ivp=detest('2OD3')
                myivp=ivp#load initial parameters from problem
                t,u = rk44(myivp)#approximate soluton of problem
                u_od_3=np.array([u[r1],u[r2],u[r3],u[r4],u[r5],u[r6],u[r7],u[r8],u[r9],u[r10], u[r11],u[r12],u[r13],u[r14],u[r15],u[r16],u[r17] ])

                ivp=detest('2OD4')
                myivp=ivp#load initial parameters from problem
                t,u = rk44(myivp)#approximate soluton of problem
                u_od_4=np.array([u[r1],u[r2],u[r3],u[r4],u[r5],u[r6],u[r7],u[r8],u[r9],u[r10], u[r11],u[r12],u[r13],u[r14],u[r15],u[r16],u[r17] ])

              #x1,x2,x3,x4 - solutions with change of 1,2,3,4 parameters with dq1,dq2,dq3,dq4 respectevely:
                x1=np.array([u_od_1[0][0],u_od_1[1][0],u_od_1[2][0],u_od_1[3][0],u_od_1[4][0],u_od_1[5][0],u_od_1[6][0],u_od_1[7][0],u_od_1[8][0],u_od_1[9][0],
                                u_od_1[10][0],u_od_1[11][0],u_od_1[12][0],u_od_1[13][0],u_od_1[14][0],u_od_1[15][0],u_od_1[16][0],
                                u_od_1[0][1],u_od_1[1][1],u_od_1[2][1],u_od_1[3][1],u_od_1[4][1],u_od_1[5][1],u_od_1[6][1],u_od_1[7][1],u_od_1[8][1],u_od_1[9][1],
                                u_od_1[10][1],u_od_1[11][1],u_od_1[12][1],u_od_1[13][1],u_od_1[14][1],u_od_1[15][1],u_od_1[16][1]])


                x2=np.array([u_od_2[0][0],u_od_2[1][0],u_od_2[2][0],u_od_2[3][0],u_od_2[4][0],u_od_2[5][0],u_od_2[6][0],u_od_2[7][0],u_od_2[8][0],u_od_2[9][0],
                                u_od_2[10][0],u_od_2[11][0],u_od_2[12][0],u_od_2[13][0],u_od_2[14][0],u_od_2[15][0],u_od_2[16][0],
                                u_od_2[0][1],u_od_2[1][1],u_od_2[2][1],u_od_2[3][1],u_od_2[4][1],u_od_2[5][1],u_od_2[6][1],u_od_2[7][1],u_od_2[8][1],u_od_2[9][1],
                                u_od_2[10][1],u_od_2[11][1],u_od_2[12][1],u_od_2[13][1],u_od_2[14][1],u_od_2[15][1],u_od_2[16][1]])

                x3=np.array([u_od_3[0][0],u_od_3[1][0],u_od_3[2][0],u_od_3[3][0],u_od_3[4][0],u_od_3[5][0],u_od_3[6][0],u_od_3[7][0],u_od_3[8][0],u_od_3[9][0],
                                u_od_3[10][0],u_od_3[11][0],u_od_3[12][0],u_od_3[13][0],u_od_3[14][0],u_od_3[15][0],u_od_3[16][0],
                                u_od_3[0][1],u_od_3[1][1],u_od_3[2][1],u_od_3[3][1],u_od_3[4][1],u_od_3[5][1],u_od_3[6][1],u_od_3[7][1],u_od_3[8][1],u_od_3[9][1],
                                u_od_3[10][1],u_od_3[11][1],u_od_3[12][1],u_od_3[13][1],u_od_3[14][1],u_od_3[15][1],u_od_3[16][1]])#solutions with with the third parameter changed by an amount dq
                x4=np.array([u_od_4[0][0],u_od_4[1][0],u_od_4[2][0],u_od_4[3][0],u_od_4[4][0],u_od_4[5][0],u_od_4[6][0],u_od_4[7][0],u_od_4[8][0],u_od_4[9][0],
                                u_od_4[10][0],u_od_4[11][0],u_od_4[12][0],u_od_4[13][0],u_od_4[14][0],u_od_4[15][0],u_od_4[16][0],
                                u_od_4[0][1],u_od_4[1][1],u_od_4[2][1],u_od_4[3][1],u_od_4[4][1],u_od_4[5][1],u_od_4[6][1],u_od_4[7][1],u_od_4[8][1],u_od_4[9][1],
                                u_od_4[10][1],u_od_4[11][1],u_od_4[12][1],u_od_4[13][1],u_od_4[14][1],u_od_4[15][1],u_od_4[16][1]])#solutions with with the 4 parameter changed by an amount dq
                

                #v1-v4 are arrays of derivatives solutions with changing parameters:
                v1=np.array([((x1[0]-x0[0])/dq1), ((x1[1]-x0[1])/dq1),((x1[2]-x0[2])/dq1), ((x1[3]-x0[3])/dq1),
                               ((x1[4]-x0[4])/dq1),  ((x1[5]-x0[5])/dq1),((x1[6]-x0[6])/dq1), ((x1[7]-x0[7])/dq1),
                               ((x1[8]-x0[8])/dq1), ((x1[9]-x0[9])/dq1),((x1[10]-x0[10])/dq1), ((x1[11]-x0[11])/dq1),
                                ((x1[12]-x0[12])/dq1), ((x1[13]-x0[13])/dq1),((x1[14]-x0[14])/dq1),((x1[15]-x0[15])/dq1),
                                  ((x1[16]-x0[16])/dq1), ((x1[17]-x0[17])/dq1),((x1[18]-x0[18])/dq1), ((x1[19]-x0[19])/dq1),             
                                    ((x1[20]-x0[20])/dq1), ((x1[21]-x0[21])/dq1),((x1[22]-x0[22])/dq1), ((x1[23]-x0[23])/dq1),
                                      ((x1[24]-x0[24])/dq1),  ((x1[25]-x0[25])/dq1),((x1[26]-x0[26])/dq1), ((x1[27]-x0[27])/dq1),
                                        ((x1[28]-x0[28])/dq1), ((x1[29]-x0[29])/dq1),((x1[30]-x0[30])/dq1), ((x1[31]-x0[31])/dq1),
                                         ((x1[32]-x0[32])/dq1), ((x1[33]-x0[33])/dq1)])#derivative with the first parameter
                v2=np.array([((x2[0]-x0[0])/dq2), ((x2[1]-x0[1])/dq2),((x2[2]-x0[2])/dq4), ((x2[3]-x0[3])/dq4),
                           ((x2[4]-x0[4])/dq2), ((x2[5]-x0[5])/dq2),((x2[6]-x0[6])/dq4), ((x2[7]-x0[7])/dq4),
                                ((x2[8]-x0[8])/dq2), ((x2[9]-x0[9])/dq2),((x2[10]-x0[10])/dq2), ((x2[11]-x0[11])/dq2),
                                 ((x2[12]-x0[12])/dq2), ((x2[13]-x0[13])/dq2),((x2[14]-x0[14])/dq2),((x2[15]-x0[15])/dq2),
                                       ((x2[16]-x0[16])/dq2), ((x2[17]-x0[17])/dq2),((x2[18]-x0[18])/dq2), ((x2[19]-x0[19])/dq2),
                                       ((x2[20]-x0[20])/dq2), ((x2[21]-x0[21])/dq2),((x2[22]-x0[22])/dq2), ((x2[23]-x0[23])/dq2),
                                             ((x2[24]-x0[24])/dq2),  ((x2[25]-x0[25])/dq2),((x2[26]-x0[26])/dq2), ((x2[27]-x0[27])/dq2),
                                              ((x2[28]-x0[28])/dq2), ((x2[29]-x0[29])/dq2),((x2[30]-x0[30])/dq2), ((x2[31]-x0[31])/dq2),
                                                ((x2[32]-x0[32])/dq2), ((x2[33]-x0[33])/dq2)])#derivative with the second parameter

                v3=np.array([((x3[0]-x0[0])/dq3), ((x3[1]-x0[1])/dq3),((x3[2]-x0[2])/dq3), ((x3[3]-x0[3])/dq3),
                             ((x3[4]-x0[4])/dq3),  ((x3[5]-x0[5])/dq3),((x3[6]-x0[6])/dq3), ((x3[7]-x0[7])/dq3),
                             ((x3[8]-x0[8])/dq3), ((x3[9]-x0[9])/dq3),((x3[10]-x0[10])/dq3), ((x3[11]-x0[11])/dq3),
                               ((x3[12]-x0[12])/dq3), ((x3[13]-x0[13])/dq3),((x3[14]-x0[14])/dq3),((x3[15]-x0[15])/dq3),
                                ((x3[16]-x0[16])/dq3), ((x3[17]-x0[17])/dq3),((x3[18]-x0[18])/dq3), ((x3[19]-x0[19])/dq3),             
                                 ((x3[20]-x0[20])/dq3), ((x3[21]-x0[21])/dq3),((x3[22]-x0[22])/dq3), ((x3[23]-x0[23])/dq3),
                                 ((x3[24]-x0[24])/dq3),  ((x3[25]-x0[25])/dq3),((x3[26]-x0[26])/dq3), ((x3[27]-x0[27])/dq3),
                                    ((x3[28]-x0[28])/dq3), ((x3[29]-x0[29])/dq3),((x3[30]-x0[30])/dq3), ((x3[31]-x0[31])/dq3),
                                    ((x3[32]-x0[32])/dq3), ((x3[33]-x0[33])/dq3)])#derivative with the first parameter
                v4=np.array([((x4[0]-x0[0])/dq4), ((x4[1]-x0[1])/dq4),((x4[2]-x0[2])/dq2), ((x4[3]-x0[3])/dq2),
                             ((x4[4]-x0[4])/dq4), ((x4[5]-x0[5])/dq4),((x4[6]-x0[6])/dq2), ((x4[7]-x0[7])/dq2),
                              ((x4[8]-x0[8])/dq4), ((x4[9]-x0[9])/dq4),((x4[10]-x0[10])/dq2), ((x4[11]-x0[11])/dq4),
                             ((x4[12]-x0[12])/dq4), ((x4[13]-x0[13])/dq4),((x4[14]-x0[14])/dq4),((x4[15]-x0[15])/dq4),
                                 ((x4[16]-x0[16])/dq4), ((x4[17]-x0[17])/dq4),((x4[18]-x0[18])/dq4), ((x4[19]-x0[19])/dq4),
                                      ((x4[20]-x0[20])/dq4), ((x4[21]-x0[21])/dq4),((x4[22]-x0[22])/dq4), ((x4[23]-x0[23])/dq4),
                                         ((x4[24]-x0[24])/dq4),  ((x4[25]-x0[25])/dq4),((x4[26]-x0[26])/dq4), ((x4[27]-x0[27])/dq4),
                                             ((x4[28]-x0[28])/dq4), ((x4[29]-x0[29])/dq4),((x4[30]-x0[30])/dq4), ((x4[31]-x0[31])/dq4),
                                                   ((x4[32]-x0[32])/dq4), ((x4[33]-x0[33])/dq4)])
                

                # derv1 -derv4 are elements of gradient:

                derv1 = -2*delta1*sum((xfact[0:17]-x0[0:17])*(v1[0:17]))-2*delta2*sum((xfact[17:]-x0[17:])*(v1[17:]))#derivative of function with the first parameter
                derv2 = -2*delta1*sum((xfact[0:17]-x0[0:17])*(v2[0:17]))-2*delta2*sum((xfact[17:]-x0[17:])*(v2[17:]))#derivative of function with the second parameter
                derv3 = -2*delta1*sum((xfact[0:17]-x0[0:17])*(v3[0:17]))-2*delta2*sum((xfact[17:]-x0[17:])*(v3[17:]))#derivative of function with the third parameter
                derv4 = -2*delta1*sum((xfact[0:17]-x0[0:17])*(v4[0:17]))-2*delta2*sum((xfact[17:]-x0[17:])*(v4[17:]))#derivative of function with the 4 parameter
                


                gradv=np.array([derv1,derv2,derv3,derv4])               

                
                norma=ln.norm(gradv)
                normlist.append(norma)
                                
                
                
       else: 
                s2=s0+a*gradv#new parameters
                s0=s2
                m=a*dr
                a=m             
    
       
            


       
          
       
       print('step of iteration')
       print(i)
       
       print('value of function which must be minimized:')
       print(func1)
       
       print('norm of gradient:')
       print(norma)
       print('results of parameters identification:')
       print('identification')
       print(s00,s01,k1,k2)
       
       print('fact value:') 
       print(xfact)
       print('solution:') 
       print(x0)       
       print('shag a:')
       print(a)
       print('total error 1:')
       print(totalpogr1)
       print('total error 2:')
       print(totalpogr2)
       print('error 1:')
       print(vectpogr1)
       print('error 2:')
       print(vectpogr2)
print('all norms')
print(normlist)
print(vectpogr1)
print('error 2:')
print(vectpogr2)
print(ilist)
print(totalpogrlist1)
maxpogr1=max(totalpogrlist1)#maximum total error
maxpogr2=max(totalpogrlist2)#maximum total error
maxpogr=max(maxpogr1,maxpogr2)



for s0 in slist:#plot for solutions with new parameters in i step
     
     
     ivp=detest('2OD')
     myivp1=ivp#load initial parameters from problem
     t,u = rk44(myivp1)#approximate soluton of problem  
     plt.axis([0,24,30,110])    
    
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
   plt.plot(i,totalpogrlist1[i],"ko",lw=0.00007)#plot for total error for 1 regions with new parameters in i step
   plt.plot(i,totalpogrlist2[i],"ro",lw=0.00007)#plot for total error for 2 regions with new parameters in i step
   
   fig2.append(plt.plot(i, totalpogrlist1[i],"ko",lw=0.00007))
   fig2.append(plt.plot(i,totalpogrlist2[i],"ro", lw=0.00007))



plt.xlabel('i')
plt.ylabel('totalpogr_i, i=1,2')
plt.legend(['total error1','total error2'], loc = 'upper right')              
plt.grid(fig2)
plt.show(fig2)



fig3=[]
plt.axis([0,24,0,5])
fig3.append(plt.plot(thlist1,vectpogr1,"ro",lw=0.007))
fig3.append(plt.plot(thlist1,vectpogr2,"go",lw=0.007))#for plot of 2 region 

plt.xlabel('t')
plt.ylabel('p')
plt.legend(['mystake for experemental value 1 and model value u1(t)', 'mystake for experemental value 2 and model value u2(t)' ], loc = 'upper right')              
plt.grid(fig3)
plt.show(fig3)



fig4=[]
plt.axis([0,max(ilist),0,normlist[0]])
for i in ilist:#plots for total error for 1 and 2 regions with new parameters in i step
   plt.plot(i,normlist[i],"ko",lw=0.00007)#plot for total error for 1 regions with new parameters in i step  
   
   fig4.append(plt.plot(i, normlist[i],"ko",lw=0.00007))
   
plt.xlabel('i')
plt.ylabel('norma')
plt.legend(['norma'], loc = 'upper right')              
plt.grid(fig4)
plt.show(fig4)


