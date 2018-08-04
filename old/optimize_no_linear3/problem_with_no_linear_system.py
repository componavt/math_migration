from decimal import Decimal 
from math import exp
import numpy as np
from numpy import *
import numpy.linalg as ln
import nodepy
from nodepy import rk
TWOPLACES = Decimal(10) ** -4
eps1 = 0.0001#accuracy
mu0=10**7 #value for changing the Hessian matrix
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

aa1=(a[0]/u0[1]+a[2]/u1[1]+a[4]/u2[1])/3.0#average coefficient of population, who arrives to region 1
aa2=(a[1]/u0[0]+a[3]/u1[0]+a[5]/u2[0])/3.0#average coefficient of population, who arrives to region 2
ad1=((d[0]/u0[0])+(d[2]/u1[0])+(d[4]/u2[0]))/3.0#average coefficient of death in region 1
ad2=((d[1]/u0[1])+(d[3]/u1[1])+(d[5]/u2[1]))/3.0#average coefficient of death in region 2

ab1=((b[0]/u0[0])+(b[2]/u1[0])+(b[4]/u2[0]))/3.0#average of born in region 1
ab2=((b[1]/u0[1])+(b[3]/u1[1])+(b[5]/u2[1]))/3.0#average of born in region 2

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
q=10**(-2)#positive parameter
p120= ((exp(q*(v[1]*exp((-z[1]*rs[0])/u0[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u0[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))#probability of transition from 1 region to 2 when time is 0         
p121= ((exp(q*(v[1]*exp((-z[1]*rs[0])/u1[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u1[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))#probability of transition from 1 region to 2 when time is 1
p122= ((exp(q*(v[1]*exp((-z[1]*rs[0])/u1[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u2[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))#probability of transition from 1 region to 2 when time is 2
p130 = ((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u0[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))#probability of transition from 1 region to 3 when time is 0
p131 = ((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u1[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))#probability of transition from 1 region to 3 when time is 1
p132 = ((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u2[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))#probability of transition from 1 region to 3 when time is 2
p210=((exp(q*(v[0]*exp((-z[1]*rs[0])/u0[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u0[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))#probability of transition from 2 region to 1 when time is 0
p211=((exp(q*(v[0]*exp((-z[1]*rs[0])/u1[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u1[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))#probability of transition from 2 region to 1 when time is 1
p212=((exp(q*(v[0]*exp((-z[1]*rs[0])/u2[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u2[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))#probability of transition from 2 region to 1 when time is 2
p230=((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u0[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))#probability of transition from 2 region to 3 when time is 0
p231=((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u1[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))#probability of transition from 2 region to 3 when time is 1
p232=((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u2[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))#probability of transition from 2 region to 3 when time is 2


s00=(mean([aa2/p120,aa2/p121,aa2/p122]))#coefficient of migration  from 1 region to 2
s01=(mean([aa1/p210,aa1/p211,aa1/p212]))#coefficient of migration  from 2 region to 1
s13=(mean([e1/p130,e1/p131,e1/p132]))
s23=(mean([e2/p230,e2/p231,e2/p232]))
s3=mean([s13,s23])#coefficient of migration  from 1 and 2 region to abroad

s0= np.array([s00,s01,s3])#vector of initial parameters


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
      ivpoptim=IVPOPTIM()
      if testkey=='2OD':#system of differential equations        
        ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
        ivpoptim.T=3.0#Final time of integration
        ivpoptim.rhs = lambda t,u: np.array([(-ad1*u[0]+ab1*u[0]-s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]-s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+c1),
                                         (-ad2*u[1]+ab2*u[1]-s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+ s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]- s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+c2)])  #right side of ODE system
        ivpoptim.dt0 = 0.01#time step
       
      elif testkey=='2OD1':# a system of differential equations with the first parameter changed by an amount dq
        ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
        ivpoptim.T=3.0#Final time of integration
        ivpoptim.rhs =lambda t,u:np.array([(-ad1*u[0]+ab1*u[0]-(s0[0]+dq)*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]-s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+c1),
                                         (-ad2*u[1]+ab2*u[1]-s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+ (s0[0]+dq)*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]- s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+c2)])
        ivpoptim.dt0 = 0.01
      elif testkey=='2OD2':#a system of differential equations with the second parameter changed by an amount dq
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u: np.array([(-ad1*u[0]+ab1*u[0]-s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+(s0[1]+dq)*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]-s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+c1),
                                         (-ad2*u[1]+ab2*u[1]-(s0[1]+dq)*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+ s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]- s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+c2)])

         ivpoptim.dt0 = 0.01
      elif testkey=='2OD3':#a system of differential equations with the third parameter changed by an amount dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u:np.array([(-ad1*u[0]+ab1*u[0]-s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]-(s0[2]+dq)*((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+c1),
                                         (-ad2*u[1]+ab2*u[1]-s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+ s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]- (s0[2]+dq)*((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+c2)]) 
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD11':#a system of differential equations with the first parameter changed by an amount 2dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u:np.array([(-ad1*u[0]+ab1*u[0]-(s0[0]+2*dq)*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]-s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+c1),
                                         (-ad2*u[1]+ab2*u[1]-s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+ (s0[0]+2*dq)*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]- s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+c2)]) 
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD12':#a system of differential equations with the first and second parameter changed by an amount dq
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u: np.array([(-ad1*u[0]+ab1*u[0]-(s0[0]+dq)*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+(s0[1]+dq)*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]-s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+c1),
                                         (-ad2*u[1]+ab2*u[1]-(s0[1]+dq)*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+ (s0[0]+dq)*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]- s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+c2)])       
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD13':#a system of differential equations with the first and third parameter changed by an amount dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u: np.array([(-ad1*u[0]+ab1*u[0]-(s0[0]+dq)*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]-(s0[2]+dq)*((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+c1),
                                         (-ad2*u[1]+ab2*u[1]-s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+ (s0[0]+dq)*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]-(s0[2]+dq)*((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+c2)])


         ivpoptim.dt0 = 0.01
      elif testkey=='2OD22':#a system of differential equations with the second parameter changed by an amount 2dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs =lambda t,u: np.array([(-ad1*u[0]+ab1*u[0]-s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+(s0[1]+2*dq)*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]-s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+c1),
                                         (-ad2*u[1]+ab2*u[1]-(s0[1]+2*dq)*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+ s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]- s0[2]*((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+c2)])
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD23':#a system of differential equations with the second and third parameter changed by an amount dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs =lambda t,u: np.array([(-ad1*u[0]+ab1*u[0]-s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+(s0[1]+dq)*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]-(s0[2]+dq)*((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+c1),
                                         (-ad2*u[1]+ab2*u[1]-(s0[1]+dq)*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+ s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]- (s0[2]+dq)*((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+c2)])

         ivpoptim.dt0 = 0.01
      
      elif testkey=='2OD33':#a system of differential equations with the third parameter changed by an amount 2dq
 
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u: np.array([(-ad1*u[0]+ab1*u[0]-s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]-(s0[2]+2*dq)*((exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]+c1),
                                         (-ad2*u[1]+ab2*u[1]-s0[1]*((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+ s0[0]*((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))/((exp(q*(v[1]*exp((-z[1]*rs[0])/u[1])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[1])/u0[2])-z[0]*rs[1])))))*u[0]- (s0[2]+2*dq)*((exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))/((exp(q*(v[0]*exp((-z[1]*rs[0])/u[0])-z[0]*rs[0])))+(exp(q*(v[2]*exp((-z[1]*rs[2])/u0[2])-z[0]*rs[2])))))*u[1]+c2)])
         ivpoptim.dt0 = 0.01
        
      else: raise Exception('Unknown Detest problem')
      ivpoptim.name=testkey
      ivpoptim.description='Problem '+testkey+' of the non-stiff DETEST suite.'
      return ivpoptim  
  

rk44 = rk.loadRKM('RK44')#load runge-kutta
ivp=detest('2OD')
myivp=ivp#load initial parameters from problem
t,u = rk44(myivp)#approximate soluton of problem   
u4=np.array([u[1667], u[3334], u[5001]])##approximate soluton in moment of time 1,2,3

ivp=detest('2OD1')
myivp2=ivp#load initial parameters from problem
t,u = rk44(myivp2)#approximate soluton of problem  
u5=np.array([u[1667], u[3334], u[5001]])

ivp=detest('2OD2')
myivp3=ivp            #load initial parameters from problem
t,u = rk44(myivp3)#approximate soluton of problem  
u6=np.array([u[1667], u[3334], u[5001]])

ivp=detest('2OD3')
myivp4=ivp#load initial parameters from problem
t,u = rk44(myivp4)#approximate soluton of problem
u7=np.array([u[1667], u[3334], u[5001]])

x0=np.array([u4[0][0],u4[1][0],u4[2][0],u4[0][1],u4[1][1],u4[2][1]])#array of solution in moment of time 1,2,3
xfact=np.array([u1[0],u2[0],u3[0],u1[1],u2[1],u3[1]])#fact values in moment of time 1,2,3
x1=np.array([u5[0][0],u5[1][0],u5[2][0],u5[0][1],u5[1][1],u5[2][1]])#solutions with with the first parameter changed by an amount dq 
x2=np.array([u6[0][0],u6[1][0],u6[2][0],u6[0][1],u6[1][1],u6[2][1]])#solutions with with the second parameter changed by an amount dq
x3=np.array([u7[0][0],u7[1][0],u7[2][0],u7[0][1],u7[1][1],u7[2][1]])##solutions with with the third parameter changed by an amount dq

func=delta1*sum((xfact[0:3]-x0[0:3])**2)+delta2*sum((xfact[3:]-x0[3:])**2)#function which must be minimized
funclist.append(func)#value of a function with initial parameters
i=0 #element number in the array of value of a functions           
print('value for changing the Hessian matrix in 0 step:')
print(mu0)
print('value of function which must be minimized in 0 step:')
print(Decimal(func).quantize(TWOPLACES))                   
v1=np.array([((x1[0]-x0[0])/dq), ((x1[1]-x0[1])/dq),((x1[2]-x0[2])/dq), ((x1[3]-x0[3])/dq),((x1[4]-x0[4])/dq), ((x1[5]-x0[5])/dq)])#derivative with the first parameter
v2=np.array([((x2[0]-x0[0])/dq), ((x2[1]-x0[1])/dq),((x2[2]-x0[2])/dq), ((x2[3]-x0[3])/dq),((x2[4]-x0[4])/dq), ((x2[5]-x0[5])/dq)])#derivative with the second parameter
v3=np.array([((x3[0]-x0[0])/dq), ((x3[1]-x0[1])/dq),((x3[2]-x0[2])/dq), ((x3[3]-x0[3])/dq),((x3[4]-x0[4])/dq), ((x3[5]-x0[5])/dq)])#derivative with the third parameter

derv1 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v1[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v1[3:]))#derivative of function with the first parameter
derv2 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v2[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v2[3:]))#derivative of function with the second parameter
derv3 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v3[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v3[3:]))#derivative of function with the third parameter
gradv=np.array([derv1,derv2,derv3])#gradient
norma=ln.norm(gradv)#norm of gradient
print('norm of gradient in 0 step:')
print(Decimal(norma).quantize(TWOPLACES))
if (norma<=eps1):
        s0=s0  
                           
while(norma>=eps1):
  
    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD11')
    myivp5=ivp#load initial parameters from problem
    t,u = rk44(myivp5)#approximate soluton of problem    
    u8=np.array([u[1667], u[3334], u[5001]])

    
    ivp=detest('2OD12')
    myivp6=ivp#load initial parameters from problem
    t,u = rk44(myivp6)#approximate soluton of problem     
    u9=np.array([u[1667], u[3334], u[5001]])
    
   
    ivp=detest('2OD13')
    myivp7=ivp#load initial parameters from problem
    t,u = rk44(myivp7)#approximate soluton of problem     
    u10=np.array([u[1667], u[3334], u[5001]])

    
    ivp=detest('2OD22')
    myivp8=ivp#load initial parameters from problem
    t,u = rk44(myivp8)#approximate soluton of problem    
    u11=np.array([u[1667], u[3334], u[5001]])
    
    ivp=detest('2OD23')
    myivp9=ivp#load initial parameters from problem
    t,u = rk44(myivp9)#approximate soluton of problem     
    u12=np.array([u[1667], u[3334], u[5001]])
    
    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD33')
    myivp10=ivp#load initial parameters from problem
    t,u = rk44(myivp10)#approximate soluton of problem       
    u13=np.array([u[1667], u[3334], u[5001]])
                              
    x11=np.array([u8[0][0],u8[1][0],u8[2][0],u8[0][1],u8[1][1],u8[2][1]])#solutions with with the first parameter changed by an amount 2dq
    x12 = np.array([u9[0][0],u9[1][0],u9[2][0],u9[0][1],u9[1][1],u9[2][1]])#solutions with with the first and second parameter changed by an amount dq 
    
    x13= np.array([u10[0][0],u10[1][0],u10[2][0],u10[0][1],u10[1][1],u10[2][1]])#solutions with with the first and third parameter changed by an amount dq 
    x22= np.array([u11[0][0],u11[1][0],u11[2][0],u11[0][1],u11[1][1],u11[2][1]])#solutions with with the second parameter changed by an amount 2dq
    
    x23= np.array([u12[0][0],u12[1][0],u12[2][0],u12[0][1],u12[1][1],u12[2][1]])#solutions with with the second and third parameter changed by an amount dq
    x33=np.array([u13[0][0],u13[1][0],u13[2][0],u13[0][1],u13[1][1],u13[2][1]])#solutions with with the third parameter changed by an amount 2dq              
    
    v11=np.array([((x11[0]-(2*x1[0]-x0[0]))/((dq)**2)), ((x11[1]-(2*x1[1]-x0[1]))/((dq)**2)),((x11[2]-(2*x1[2]-x0[2]))/((dq)**2)),
                           ((x11[3]-(2*x1[3]-x0[3]))/((dq)**2)),((x11[4]-(2*x1[4]-x0[4]))/((dq)**2)), ((x11[5]-(2*x1[5]-x0[5]))/((dq)**2))])#second derivative with 1 parameter of function
         
    v21=np.array([((x12[0]-x1[0]-x2[0]+x0[0])/((dq)**2)), ((x12[1]-x1[1]-x2[1]+x0[1])/((dq)**2)),((x12[2]-x1[2]-x2[2]+x0[2])/((dq)**2)),
                           ((x12[3]-x1[3]-x2[3]+x0[3])/((dq)**2)),((x12[4]-x1[4]-x2[4]+x0[4])/((dq)**2)), ((x12[5]-x1[5]-x2[5]+x0[5])/((dq)**2))])#derivative with 1 and 2 parameter of function 
               
    v31=np.array([((x13[0]-x1[0]-x3[0]+x0[0])/((dq)**2)), ((x13[1]-x1[1]-x3[1]+x0[1])/((dq)**2)),((x13[2]-x1[2]-x3[2]+x0[2])/((dq)**2)),
                           ((x13[3]-x1[3]-x3[3]+x0[3])/((dq)**2)),((x13[4]-x1[4]-x3[4]+x0[4])/((dq)**2)), ((x13[5]-x1[5]-x3[5]+x0[5])/((dq)**2))])#derivative with 1 and 3 parameter of function
           
    v22=np.array([((x22[0]-(2*x2[0]-x0[0]))/((dq)**2)), ((x22[1]-(2*x2[1]-x0[1]))/((dq)**2)),((x22[2]-(2*x2[2]-x0[2]))/((dq)**2)),
                           ((x22[3]-(2*x2[3]-x0[3]))/((dq)**2)),((x22[4]-(2*x2[4]-x0[4]))/((dq)**2)), ((x22[5]-(2*x2[5]-x0[5]))/((dq)**2))])#second derivative with 2 parameter of function
           
    v23=np.array([((x23[0]-x3[0]-x2[0]+x0[0])/((dq)**2)), ((x23[1]-x3[1]-x2[1]+x0[1])/((dq)**2)),((x23[2]-x3[2]-x2[2]+x0[2])/((dq)**2)),
                           ((x23[3]-x3[3]-x2[3]+x0[3])/((dq)**2)),((x23[4]-x3[4]-x2[4]+x0[4])/((dq)**2)), ((x23[5]-x3[5]-x2[5]+x0[5])/((dq)**2))])#derivative with 2 and 3 parameter of function
         
    v33=np.array([((x33[0]-(2*x3[0]-x0[0]))/((dq)**2)), ((x33[1]-(2*x3[1]-x0[1]))/((dq)**2)),((x33[2]-(2*x3[2]-x0[2]))/((dq)**2)),
                              ((x33[3]-(2*x3[3]-x0[3]))/((dq)**2)),((x33[4]-(2*x3[4]-x0[4]))/((dq)**2)), ((x33[5]-(2*x3[5]-x0[5]))/((dq)**2))])#second derivative with 3 parameter of function

    #Construct the Hessian matrix:
    derv11=delta1*sum(2*(v1[0:3])**2-2*(xfact[0:3]-x0[0:3])*v11[0:3])+delta2*sum(2*(v1[3:])**2-2*(xfact[3:]-x0[3:])*v11[3:])
    derv12 =delta1*sum(2*(v1[0:3])*v2[0:3]-2*(xfact[0:3]-x0[0:3])*v21[0:3])+delta2*sum(2*(v1[3:])*v2[3:]-2*(xfact[3:]-x0[3:])*v21[3:])
    derv13=delta1*sum(2*(v1[0:3])*v3[0:3]-2*(xfact[0:3]-x0[0:3])*v31[0:3])+delta2*sum(2*(v1[3:])*v3[3:]-2*(xfact[3:]-x0[3:])*v31[3:])
    derv21 =delta1*sum(2*(v1[0:3])*v2[0:3]-2*(xfact[0:3]-x0[0:3])*v21[0:3])+delta2*sum(2*(v1[3:])*v2[3:]-2*(xfact[3:]-x0[3:])*v21[3:])
    derv22=delta1*sum(2*(v2[0:3])**2-2*(xfact[0:3]-x0[0:3])*v22[0:3])+delta2*sum(2*(v2[3:])**2-2*(xfact[3:]-x0[3:])*v22[3:])
    derv23=delta1*sum(2*(v2[0:3])*v3[0:3]-2*(xfact[0:3]-x0[0:3])*v23[0:3])+delta2*sum(2*(v2[3:])*v3[3:]-2*(xfact[3:]-x0[3:])*v23[3:])
    derv31 =delta1*sum(2*(v3[0:3])*v1[0:3]-2*(xfact[0:3]-x0[0:3])*v31[0:3])+delta2*sum(2*(v3[3:])*v1[3:]-2*(xfact[3:]-x0[3:])*v31[3:])
    derv32 =delta1*sum(2*(v3[0:3])*v2[0:3]-2*(xfact[0:3]-x0[0:3])*v23[0:3])+delta2*sum(2*(v3[3:])*v2[3:]-2*(xfact[3:]-x0[3:])*v23[3:])
    derv33=delta1*sum(2*(v3[0:3])**2-2*(xfact[0:3]-x0[0:3])*v33[0:3])+delta2*sum(2*(v3[3:])**2-2*(xfact[3:]-x0[3:])*v33[3:])
           
    hesse=np.array([[derv11,derv12,derv13],[derv21,derv22,derv23],[derv31,derv32,derv33]])#The Matrix of Hesse
    #print(hesse)
    E=np.diag((1.,1.,1.))#identity matrix
    chhesse=hesse+mu0*E#modified Hessian matrix
    inversechhesse=np.linalg.inv(chhesse)# inverse matrix of modified Hessian matrix    
    p=np.dot(inversechhesse,(gradv.T))#vector for calculating  new parameters   
    s1=s0-p.T#new parameters
    s0=s1
    
    
    rk44 = rk.loadRKM('RK44')#load runge-kutta, find solutions with new parameters
    ivp=detest('2OD')
    myivp=ivp#load initial parameters from problem
    t,u = rk44(myivp)#approximate soluton of problem      
    u4=np.array([u[1667], u[3334], u[5001]])
    x0=np.array([u4[0][0],u4[1][0],u4[2][0],u4[0][1],u4[1][1],u4[2][1]])#solutions with new parameters in time 1,2,3        
    func1=delta1*sum((xfact[0:3]-x0[0:3])**2)+delta2*sum((xfact[3:]-x0[3:])**2)#Find value of function with new parameters    
    funclist.append(func1)#value of a function with initial parameters       
    i=i+1 #element number in the array of value of a functions with new parameters
                    
    #compare the new and previous values of the function:          
    if (funclist[i-1]> funclist[i]):
               
                s0=s1#take new values of the parameters
                mu=(mu0)/2#halve the step
                mu0=mu
                #Find gradien and norm of gradient and return to the beginning of the cycle to calculate the Hesse matrix      
                rk44 = rk.loadRKM('RK44')#load runge-kutta
                ivp=detest('2OD1')
                myivp2=ivp#load initial parameters from problem
                t,u = rk44(myivp2)#approximate soluton of problem             
                u5=np.array([u[1667], u[3334], u[5001]])
               
                ivp=detest('2OD2')
                myivp3=ivp#load initial parameters from problem
                t,u = rk44(myivp3)#approximate soluton of problem              
                u6=np.array([u[1667], u[3334], u[5001]])

                
                ivp=detest('2OD3')
                myivp4=ivp#load initial parameters from problem
                t,u = rk44(myivp4)#approximate soluton of problem               
                u7=np.array([u[1667], u[3334], u[5001]])
                
                x1=np.array([u5[0][0],u5[1][0],u5[2][0],u5[0][1],u5[1][1],u5[2][1]])#solutions with with the first parameter changed by an amount dq
                x2=np.array([u6[0][0],u6[1][0],u6[2][0],u6[0][1],u6[1][1],u6[2][1]])#solutions with with the second parameter changed by an amount dq
                x3=np.array([u7[0][0],u7[1][0],u7[2][0],u7[0][1],u7[1][1],u7[2][1]])#solutions with with the third parameter changed by an amount dq
   

                v1=np.array([((x1[0]-x0[0])/dq), ((x1[1]-x0[1])/dq),((x1[2]-x0[2])/dq), ((x1[3]-x0[3])/dq),((x1[4]-x0[4])/dq), ((x1[5]-x0[5])/dq)])#derivative with the first parameter
                v2=np.array([((x2[0]-x0[0])/dq), ((x2[1]-x0[1])/dq),((x2[2]-x0[2])/dq), ((x2[3]-x0[3])/dq),((x2[4]-x0[4])/dq), ((x2[5]-x0[5])/dq)])#derivative with the second parameter
                v3=np.array([((x3[0]-x0[0])/dq), ((x3[1]-x0[1])/dq),((x3[2]-x0[2])/dq), ((x3[3]-x0[3])/dq),((x3[4]-x0[4])/dq), ((x3[5]-x0[5])/dq)])#derivative with the third parameter
                
                derv1 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v1[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v1[3:]))#derivative of function with the first parameter
                derv2 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v2[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v2[3:]))#derivative of function with the second parameter)
                derv3 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v3[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v3[3:]))#derivative of function with the third parameter
                gradv=np.array([derv1,derv2,derv3])#gradient                
                norma=ln.norm(gradv)#norm of gradient
                s0=s1   
    if (funclist[i-1]<=funclist[i]):
              mu=(mu0)*2#double the step
              mu0=mu   
              s0=s0+p.T#take the previous values of the parameters and return to the beginning of the cycle to recalculate the Hesse matrix   
              
    s00=round(s0[0],4)
    s01=round(s0[1],4)
    s3=round(s0[2],4)
    
    print('value for changing the Hessian matrix or step:')
    print(mu0)
    print('value of function which must be minimized:')
    print(Decimal(func1).quantize(TWOPLACES))
    print('norm of gradient:')
    print(Decimal(norma).quantize(TWOPLACES))
    print('results of parameters identification:')
    print(s00, s01,s3)
                 
             
