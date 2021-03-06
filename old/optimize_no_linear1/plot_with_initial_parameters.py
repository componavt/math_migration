from math import exp
import numpy as np
from numpy import *
import numpy.linalg as ln
from matplotlib import mlab
import matplotlib.pyplot as plt
import pylab
import nodepy
from nodepy import rk
import nodepy.linear_multistep_method as lm
eps1 = 0.0001#accuracy
a = np.array([20.,10.,15.,43.,25.,45. ])#arrive to region,when time is 0,1,2 in 1 and 2 region respectively
l=np.array([10.,20.,43.,15.,45.,25.])#leave region, time is 0,1,2 from 1 and 2 region respectively 
b = np.array([10.,10.,17.,10.,43.,23.])#born in regoin 1 and 2 respectively,time is 0,1,2  
d = np.array([15.,25.,25.,40.,25.,16.])#death in region 1 and 2 respectively , time is 0,1,2
ga= np.array([5.,7.,2.,3.,3.,4. ])#go to abroad from region 1 and 2 respectively , time is 0,1,2
fa =np.array([3.,3.,2.,6.,4.,7.])#g0 from abroad to region 1 and 2 respectively , time is 0,1,2 

u0=np.array([100.,70.,200.])#initial value of population in 1 and 2 region respectively,third parameter is constant population abroad 
u1= np.array([(u0[0]-d[0]+b[0]+a[0]-l[0]-ga[0]+fa[0]),(u0[1]-d[1]+b[1]+a[1]-l[1]-ga[1]+fa[1])])#numer of population, time 1
u2= np.array([(u1[0]-d[2]+b[2]+a[2]-l[2]-ga[2]+fa[2]),(u1[1]-d[3]+b[3]+a[3]-l[3]-ga[3]+fa[3])])#numer of population, time 2
u3= np.array([(u2[0]-d[4]+b[4]+a[4]-l[4]-ga[4]+fa[4]),(u2[1]-d[5]+b[5]+a[5]-l[5]-ga[5]+fa[5])])#numer of population, time 3

aa1=(a[0]/u0[1]+a[2]/u1[1]+a[4]/u2[1])/3.0#average number of population, who arrives to region 1
aa2=(a[1]/u0[0]+a[3]/u1[0]+a[5]/u2[0])/3.0#average number of population, who arrives to region 2
ad1=((d[0]/u0[0])+(d[2]/u1[0])+(d[4]/u2[0]))/3.0#average number of death in region 1
ad2=((d[1]/u0[1])+(d[3]/u1[1])+(d[5]/u2[1]))/3.0#average number of death in region 2
#print(aa1, aa2)
#al1=((l[0]/u0[0])+(l[2]/u1[0])+(l[4]/u2[0]))/3.0#average number of population, who leaves region 1
#al2=((l[1]/u0[1])+(l[3]/u1[1])+(l[5]/u2[1]))/3.0#average number of population, who leaves region 2
#print(aa1)
#print(aa2)
#print(al1)
#print(al2)
ab1=((b[0]/u0[0])+(b[2]/u1[0])+(b[4]/u2[0]))/3.0#average of born in region 1
ab2=((b[1]/u0[1])+(b[3]/u1[1])+(b[5]/u2[1]))/3.0#average of born in region 2
aa12=aa1/u0[1]
aa22=aa2/u0[0]
c1= (fa[0]+fa[2]+fa[4])/3.0  #average number population, who goes from abroad to region 1
c2=(fa[1]+fa[3]+fa[5])/3.0#average number population, who goes from abroad to region 2
v=np.array([0.9,1.2,1.4])# true utility of 1 region, 2 regions and abroad
z=np.array([0.8,0.3])#parameters alpha, betta
rs=np.array([100.,200.,300.])#distance between 1 and 2 region, between abroad and 1 region, between 2 region and abroad
e1= ((ga[0]/u0[0])+(ga[2]/u1[0])+(ga[4]/u2[0]))/3.0#average number population, who goes to abroad from region 1 and 2
e2=((ga[1]/u0[1])+(ga[3]/u1[1])+(ga[5]/u2[1]))/3.0
#s0= np.array([aa22,aa12,e1])#vector of initial parameters
#print(aa22, aa12)
#print(e1)
dq=10**(-4)
funclist=[ ]
t = np.linspace(0.,3.)
delta1=0.5#weight coefficient for function, which must be minimized,delta1+delta2=1
delta2=0.5#weight coefficient for function, which must be minimized,delta1+delta2=1
q=10**(-2)
p120= ((exp(q*(v[1]*exp((-z[1]*rs[0])/u0[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u0[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))
p121= ((exp(q*(v[1]*exp((-z[1]*rs[0])/u1[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u1[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))
p122= ((exp(q*(v[1]*exp((-z[1]*rs[0])/u1[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u2[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))
p130 = ((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u0[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))
p131 = ((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u1[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))
p132 = ((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u2[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))
p210=((exp(q*(v[0]*exp((-z[1]*rs[0])/u0[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u0[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))
p211=((exp(q*(v[0]*exp((-z[1]*rs[0])/u1[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u1[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))
p212=((exp(q*(v[0]*exp((-z[1]*rs[0])/u2[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u2[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))
p230=((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u0[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))
p231=((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u1[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))
p232=((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u2[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))


s00=(mean([aa2/p120,aa2/p121,aa2/p122]))
s01=(mean([aa1/p210,aa1/p211,aa1/p212]))
s13=(mean([e1/p130,e1/p131,e1/p132]))
s23=(mean([e2/p230,e2/p231,e2/p232]))
s3=mean([s13,s23])
s0=([s00,s01,s3])#vector of initial parameters
class IVPOPTIM(object):  #Problems that are solved by the Runge-Kutta method
     def __init__(self, f=None, u0=np.array([100.,70.]), t0=0., T=3.0, dt0=0.01, exact=None, desc='', name=''):
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
      ivpoptim=IVPOPTIM3111()
      if testkey=='2OD':#system of differential equations        
        ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
        ivpoptim.T=3.0#Final time of integration
        ivpoptim.rhs = lambda t,u: np.array([(-ad1*u[0]+ab1*u[0]-s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]-s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+c1),
                                         (-ad2*u[1]+ab2*u[1]-s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+ s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]- s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+c2)])  #right side of ODE system
        ivpoptim.dt0 = 0.01#time step
      else: raise Exception('Unknown Detest problem')
      ivpoptim.name=testkey
      ivpoptim.description='Problem '+testkey+' of the non-stiff DETEST suite.'
      return ivpoptim   


rk44 = rk.loadRKM('RK44')#load runge-kutta
ivp=detest('2OD')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem
#u4=np.array([u[1667], u[3334], u[5001]])
#t4=np.array([t[1667], t[3334], t[5001]])
#print(u4)

t1 = np.array([0.0, 1.0, 2.0, 3.0])#time for fact value
z1 = np.array([u0[0], u1[0], u2[0],u3[0]])#fact value
z2 = np.array([u0[1], u1[1], u2[1],u3[1]])#fact value
plt.axis([0, 5,0,200])
plt.plot(t, u, )#model solution
plt.plot(t1, z1, "r--",marker='o' )#fact value
plt.plot(t1, z2, "g--",marker='o' )#fact value

plt.xlabel('t')
plt.ylabel('u')
plt.legend(['x(t)', 'y(t)','z1(t)','z2(t)'], loc = 'central')
plt.grid()
plt.show()
















