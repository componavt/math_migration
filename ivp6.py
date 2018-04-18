from __future__ import print_function
from __future__ import absolute_import
import numpy as np
class IVP6(object):    
    def __init__(self, f=None, u0=1., t0=0., T=3.0, exact=None, desc='', name=''):
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
m = np.array([20.,10.])#arrive to region, time is 0 
w=np.array([10.,20.])#leave region, time 0
b = np.array([10.,10.])#born in regoin, time 0
sm = np.array([15.,25.])#death in region, time 0
l= np.array([35.,15.])#go to abroad, time 0
k =np.array([40.,20.])#g0 from abroad, time 0

u0=np.array([100.,70.])#initial value
u1= np.array([(u0[0]-sm[0]+b[0]+m[0]-w[0]-l[0]+k[0]),(u0[1]-sm[1]+b[1]+m[1]-w[1]-l[1]+k[1])])#numer of population, time 1
m1 = np.array([15.,43.])#arrive to region,time 1
w1=np.array([43.,15.])#leave region, time 1
b1 = np.array([17.,10.])#born in regoin, time 1
sm1 = np.array([25.,40.])#death in region,time 1
l1= np.array([39.,43.])#go to abroad, time 1
k1=np.array([35.,5.])#g0 from abroad, time 1
u2= np.array([(u1[0]-sm1[0]+b1[0]+m1[0]-w1[0]-l1[0]+k1[0]),(u1[1]-sm1[1]+b1[1]+m1[1]-w1[1]-l1[1]+k1[1])])#numer of population, time 2
m2 = np.array([25.,45.])#arrive to region,time 2
w2=np.array([45.,25.])#leave region, time 2
b2 = np.array([43.,23.])#born in regoin, time 2
sm2 = np.array([25.,16.])#death in region,time 2
l2= np.array([30.,10.])#go to abroad, time 2
k2= np.array([35.,25.])#g0 from abroad, time 2
u3= np.array([(u2[0]-sm2[0]+b2[0]+m2[0]-w2[0]-l2[0]+k2[0]),(u2[1]-sm2[1]+b2[1]+m2[1]-w2[1]-l2[1]+k2[1])])#numer of population, time 3
g = (m[0]+m[1]+m1[0]+m1[1]+m2[0]+m2[1])/6.0 #average number of population, who arrives to region 
a =(sm[0]+sm[1]+sm1[0]+sm1[1]+sm2[0]+sm2[1])/6.0#average number of death in region
d=(w[0]+w[1]+w1[0]+w1[1]+w2[0]+w2[1])/6.0#average number of population, who leaves region
r=(b[0]+b[1]+b1[0]+b1[1]+b2[0]+b2[1])/6.0#average of born in region 
c=(k[0]+k[1]+k1[0]+k1[1]+k2[0]+k2[1])/6.0#average number population, who goes from abroad
s =(l[0]+l[1]+l1[0]+l1[1]+l2[0]+l2[1])/6.0 #verage number population, who goes to abroad
a1= np.array([(a/u0[0]),(a/u0[1])])
b1 = np.array([(r/u0[0]),(r/u0[1])])
c1=np.array([(d/u0[0]),(d/u0[1])])
d1 = np.array([(g/u0[1]),(g/u0[0])])
e1= np.array([(s/u0[1]),(s/u0[0])])

print(u0,u1,u2,u3)
def detest(testkey):#Description of problem
    import numpy as np
    ivp6=IVP6()
    if testkey=='2OD':        
        ivp6.u0=np.array([u0[0],u0[1]])#initial parameters
        ivp6.T=3.0#Final time of integration
        ivp6.rhs = lambda t,u: np.array([(-a1[0]*u[0]+b1[0]*u[0]-c1[0]*u[0]+d1[0]*u[1]-e1[0]*u[0]+c),(-a1[1]*u[1]+b1[1]*u[1]-c1[1]*u[1]+d1[1]*u[0]-e1[1]*u[1]+c)]) #right side of ODE system
        ivp6.dt0 = 0.01#time step
    else: raise Exception('Unknown Detest problem')
    ivp6.name=testkey
    ivp6.description='Problem '+testkey+' of the non-stiff DETEST suite.'
    return ivp6
