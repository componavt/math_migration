import math
import numpy as np
from scipy.integrate import odeint
from matplotlib import mlab
import matplotlib.pyplot as plt

#model of DE
def model(w,t):
    x = w[0]
    y = w[1]
    dxdt =274049.3333-(2.2/1000)*x-(151971/38819900)*x+((57173-50659)/(38819900*13800700))*x*y
    dydt =119359.3333-(1/1000)*y-(75557/13800700)*y-((57173-50659)/(38819900*13800700))*x*y
    return [dxdt, dydt]
w0=[38819900.0,13800700.0]#initial parameter

# right part of the equation
def eq (p,t):
    l = p[0]
    m = p[1]
    l =274049.3333 -(2.2/1000)*x-(151971/38819900)*x+((57173-50659)/(38819900*13800700))*x*y
    m = 119359.3333-(1/1000)*y-(75557/13800700)*y-((57173-50659)/(38819900*13800700))*x*y
    return [l,m]

# value of right part with initial parameter w0
x0=38819900.0
y0 =13800700.0

l0 =274049.3333 -(2.2/1000)*x0-(151971/38819900)*x0+((57173-50659)/(38819900*13800700))*x0*y0
m0 =119359.3333 -(1/1000)*y0-(75557/13800700)*y0-((57173-50659)/(38819900*13800700))*x0*y0
p0 = [l0,m0]
#print('value of right part with initial parameter w0, [l0, m0]  = ')
print(p0)


t0 = 0 #2014
tn = 3 # 2017
h = 0.01#time step
ilist = range(1,4,1)#array of index (1-3) for time step, (for Euler)
tlist = [(t0+h*i) for i in ilist]#value of points in time interval(0,3) (for Euler)
prev1 = x0
prev2 = y0
#t = np.linspace(t0,tn)
jlist = range(5,301,1)#array of index (5-300) for time step, (for Adams-Bashforth )
thlist = [(t0+h*j) for j in jlist]#Value of points in time interval(0,3),(for Adams-Bashforth)

#Euler
for t  in tlist:
   x = prev1+h*l0
   prev1 = x
   y = prev2+h*m0
   prev2 = y
   print(x,y)
#print(x,y)
x1 = 38821786.455533#first solution (Euler)
y1 = 13801893.593333#first solution (Euler)
#value of right part with  first solution
l1 =274049.3333-(2.2/1000)*x1-(151971/38819900)*x1+((57173-50659)/(38819900*13800700))*x1*y1
m1 =119359.3333-(1/1000)*y1-(75557/13800700)*y1-((57173-50659)/(38819900*13800700))*x1*y1
p1 = [l1, m1]
#print('value of right part with first solution , [l1, m1]  = ')
#print(p1)
x2 = 38823672.911065996#second solurion (Euler)
y2 = 13803087.186666 #second solution (Euler)
#value of right part with  second solution
l2 =274049.3333 -(2.2/1000)*x2-(151971/38819900)*x2+((57173-50659)/(38819900*13800700))*x2*y2
m2 =119359.3333 -(1/1000)*y2-(75557/13800700)*y2-((57173-50659)/(38819900*13800700))*x2*y2

p2 = [l2, m2]
#print('value of right part with second solution, [l2, m2]  = ')
#print(p2)
x3 = 38825559.36659899#third solurion (Euler)
y3 = 13804280.779999001#third solurion (Euler)
#value of right part with  third solution
l3 =274049.3333-(2.2/1000)*x3-(151971/38819900)*x3+((57173-50659)/(38819900*13800700))*x3*y3
m3 =119359.3333-(1/1000)*y3-(75557/13800700)*y3-((57173-50659)/(38819900*13800700))*x3*y3
p3 = [l3, m3]
#print('value of right part with  , [l3, m3]  = ')
#print(p3)
xhvost = (55/24)*l3-(59/24)*l2+(37/24)*l1-(9/24)*l0#right part of Adams-Bashforth formula for solution x4 without x3
#print(xhvost)
x4 = x3+h*xhvost# fourth solution
#print(x4)
yhvost = (55/24)*m3-(59/24)*m2+(37/24)*m1-(9/24)*m0#right part of Adams-Bashforth formula for solution y4 without y3
#print(yhvost)
y4 = y3+h*yhvost# fourth solution
#print(y4)
l4 =274049.3333 -(2.2/1000)*x4-(151971/38819900)*x4+((57173-50659)/(38819900*13800700))*x4*y4#value of right part with  fourth solution
m4 =119359.3333 -(1/1000)*y4-(75557/13800700)*y4-((57173-50659)/(38819900*13800700))*x4*y4#value of right part with  fourth solution
p4 = [l4, m4]
#print('value of right part with fourth solution , [l4, m4]  = ')
#print(p4)
prew1 = x4
prew2 = y4

fun10 = l0
fun20 = m0
fun11 = l1
fun21 = m1
fun12 = l2
fun22 = m2
fun23 = m3
fun13 = l3
fun14 = l4
fun24 = m4
xlist = []
ylist = []
zlist = []
#Adams-Bashforth
for t in thlist:
   x = prew1 + h*((55/24)*fun14-(59/24)*fun13+(37/24)*fun12-(9/24)*fun11)
   y = prew2+h*((55/24)*fun24-(59/24)*fun23+(37/24)*fun22-(9/24)*fun21)

   fun11 = fun12
   fun12 = fun13
   fun13 = fun14
   fun14 =274049.3333 -(2.2/1000)*x-(151971/38819900)*x+((57173-50659)/(38819900*13800700))*x*y
   fun21 = fun22
   fun22 = fun23
   fun23 = fun24
   fun24 =119359.3333  -(1/1000)*y-(75557/13800700)*y-((57173-50659)/(38819900*13800700))*x*y 
   prew1 = x
   prew2 = y
   xlist.append(prew1)# array of solution for function graph
   ylist.append(prew2)# array of solution for function graph

   #print(fun11,fun12,fun13,fun14)

   #print(fun21,fun22,fun23,fun24)

   #print(fun31,fun32,fun33,fun34)
   print(x,y)
#function graph   
plt.plot(thlist, xlist, "b-",)
plt.plot(thlist, ylist, "g-",)
plt.legend()
plt.grid()
plt.show()   
