from __future__ import print_function
from __future__ import absolute_import
import numpy as np
class IVPTEST(object):    
    def __init__(self, f=None, u0=1., t0=0., T=2.0, exact=None, desc='', name=''):
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
    ivptest=IVPTEST()    
    if testkey=='2OD1':
        ivptest.u0=np.array([1.,3.])#initial value
        ivptest.T=20.#final time
        ivptest.rhs = lambda t,u: np.array([2.*(u[0]-u[0]*u[1]),-(u[1]-u[0]*u[1])])#right side of model
        ivptest.dt0 = 1.e-2#step
    else: raise Exception('Unknown Detest problem')
    ivptest.name=testkey
    ivptest.description='Problem '+testkey+' of the non-stiff DETEST suite.'
    return ivptest
