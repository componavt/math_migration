import math
import numpy as np
from scipy.integrate import odeint
from matplotlib import mlab
import matplotlib.pyplot as plt
import pylab
#model of DE
#def model(w,t):
    #x = w[0]
    #y = w[1]
    #dxdt = 40.0+((10.0-15.0)/100.0)*x-(35.0/100.0)*x+((20.0-10.0)/(100.0*70.0))*x*y
   # dydt =20.0+((10.0-25.0)/70.0)*y-(15.0/70.0)*y+((10.0-20.0)/(70.0*100.0))*x*y 
   # return [dxdt, dydt]
#w0=[100.0,70.0]


# value of right part with initial parameter w0
x0 = 100.0
y0 =70.0
l0 = 40.0+((10.0-15.0)/100.0)*x0-(35.0/100.0)*x0+((20.0-10.0)/(100.0*70.0))*x0*y0
m0 = 20.0+((10.0-25.0)/70.0)*y0-(15.0/70.0)*y0+((10.0-20.0)/(70.0*100.0))*x0*y0

#Euler
t0 = 0 #2011
tn = 3 # 2014
h = 0.01

ilist = range(1,4,1)
#print(ilist)
tlist = [(t0+h*i) for i in ilist]
#print(tlist)

prev1 = x0
prev2 = y0
#t = np.linspace(t0,tn)
jlist = range(4,301,1)
print(jlist)
thlist = [(t0+h*j) for j in jlist]
#print(thlist)
for t  in tlist:
   x = prev1+h*l0
   prev1 = x
   y = prev2+h*m0
   prev2 = y
   print(x,y)  

x1 =100.1 #first solution (Euler)
y1 =69.8  #first solution (Euler)
#value of right part with  first solution
l1 = 40.0+((10.0-15.0)/100.0)*x1-(35.0/100.0)*x1+((20.0-10.0)/(100.0*70.0))*x1*y1
m1= 20.0+((10.0-25.0)/70.0)*y1-(15.0/70.0)*y1+((10.0-20.0)/(70.0*100.0))*x1*y1
x2 = 100.19999999999999    #second solurion (Euler)
y2 = 69.6 #second solution (Euler)
#value of right part with  second solution
l2 = 40.0+((10.0-15.0)/100.0)*x2-(35.0/100.0)*x2+((20.0-10.0)/(100.0*70.0))*x2*y2
m2 = 20.0+((10.0-25.0)/70.0)*y2-(15.0/70.0)*y2-((20.0-10.0)/(70.0*100.0))*x2*y2
x3 =100.29999999999998  #third solurion (Euler)
y3 =69.39999999999999  #third solurion (Euler)
#value of right part with  third solution
l3 = 40.0+((10.0-15.0)/100.0)*x3-(35.0/100.0)*x3+((20.0-10.0)/(100.0*70.0))*x3*y3
m3 = 20.0+((10.0-25.0)/70.0)*y3-(15.0/70.0)*y3-((20.0-10.0)/(70.0*100.0))*x3*y3
prew1 = x3
prew2 = y3
fun10 = l0
fun20 = m0
fun11 = l1
fun21 = m1
fun12 = l2
fun22 = m2
fun23 = m3
fun13 = l3
xlist = []
ylist = []
#print(fun11,fun12,fun13,fun14)
for t in thlist:
   x = prew1 + h*((55.0/24.0)*fun13-(59.0/24.0)*fun12+(37.0/24.0)*fun11-(9.0/24.0)*fun10)
   y = prew2+h*((55.0/24.0)*fun23-(59.0/24.0)*fun22+(37.0/24.0)*fun21-(9.0/24.0)*fun20)
   fun10 = fun11
   fun11 = fun12
   fun12 = fun13
   fun13 =40.0+((10.0-15.0)/100.0)*x-(35.0/100.0)*x+((20.0-10.0)/(100.0*70.0))*x*y
   fun20 = fun21
   fun21 = fun22
   fun22= fun23
   fun23 =20.0+((10.0-25.0)/70.0)*y-(15.0/70.0)*y-((20.0-10.0)/(70.0*100.0))*x*y
   prew1 = x
   prew2 = y
   xlist.append(prew1)# array of solution for function graph
   ylist.append(prew2)# array of solution for function graph
   print(x,y)
# points of fact value for function graph:
#N = 3
#t1 = np.arange(0, N+1, 1)
t1 = np.array([0.0, 1.0, 2.0, 3.0]) 
z1 = np.array([100.0, 110.0, 70.0,73.0])
z2 = np.array([70.0, 50.0, 10.0,52.0])#for j in jlist:

      
#print(ylist[96])
#print(ylist[196])
#print(ylist[296])
ylist2 = np.array([ylist[96],ylist[196],ylist[296]])
#print(ylist2)
#print(thlist[96])
thlist2=np.array([thlist[96],thlist[196],thlist[296]])
xlist2 = np.array([xlist[96],xlist[196],xlist[296]])

#function graph:
#thqlist = mlab.frange (a, b, dx)
#print(thqlist)
t2=np.arange(0,4,1)
plt.axis([0, 5,0,120]) 
plt.plot(thlist, xlist, "b-", )#model solution
plt.plot(thlist, ylist, "g-", )#model solution
plt.plot(t1, z1, "b--",marker='o' )#fact value
plt.plot(t1, z2, "g--",marker='o' )#fact value
plt.plot(thlist2, ylist2, 'go')
plt.plot(thlist2, xlist2, 'bo')
plt.xlabel('t')
plt.ylabel('y')
plt.legend(['x(t)', 'y(t)','z1(t)','z2(t)'], loc = 'central')
plt.grid()
plt.show()   
   
  
