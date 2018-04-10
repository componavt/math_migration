# Who
# What 
# ##############################

from __future__ import print_function
from __future__ import absolute_import
import numpy as np
class IVP4(object):    
    def __init__(self, f=None, u0=1., t0=0., T=1., exact=None, desc='', name=''):
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
    ivp4=IVP4()
    if testkey=='2OD':
        ivp4.u0=np.array([100.,70.])#initial parameters
        ivp4.T=3.0#Final time of integration
        ivp4.rhs = lambda t,u: np.array([( -0.15*u[0]+0.1*u[0]-0.1*u[0]+(2.0/7.0)*u[1]-0.35*u[0]+((40.0+20.0+35.0+5.0+35.0+25.0)/6.0)) ,((-5.0/14.0)*u[1]+(1.0/7.0)*u[1]-(2.0/7.0)*u[1]+(1.0/10.0)*u[0]-(3.0/14.0)*u[1]+((40.0+20.0+35.0+5.0+35.0+25.0)/6.0))]) #right side of ODE system
        ivp4.dt0 = 0.01#time step
    else: raise Exception('Unknown Detest problem')
    ivp4.name=testkey
    ivp4.description='Problem '+testkey+' of the non-stiff DETEST suite.'
    return ivp4
