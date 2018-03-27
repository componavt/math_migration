
from nodepy import ivp
import nodepy.linear_multistep_method as lm
from nodepy import ode_solver
def migration (ivp, u0,T, rhs,dt0):
        ivp.u0=np.array([100.0,70.0])#initial parametrs
        ivp.T=3.0#Final time
        #rigt side of equations
        ivp.rhs = lambda t,u: np.array([40.0+((10.0-15.0)/100.0)*u[0]-(35.0/100.0)*u[0]+((20.0-10.0)/(100.0*70.0))*u[0]*u[1],20.0+((10.0-25.0)/70.0)*u[1]-(15.0/70.0)*u[1]+((10.0-20.0)/(70.0*100.0))*u[0]*u[1]])
        ivp.dt0 = 0.01#time step
        u0=ivp.u0
        T=ivp.T
        f=ivp.rhs
        dt0 = ivp.dt0
        return(f,u0,T,dt0) 
ab4=lm.Adams_Bashforth(4)
t, u = ab4(migration)

