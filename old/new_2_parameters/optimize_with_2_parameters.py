 
from math import exp
import numpy as np
from numpy import *
import numpy.linalg as ln
import nodepy
from nodepy import rk
eps1 = 0.1#accuracy
mu0=10**4 #value for changing the Hessian matrix
a = np.array([20.,10.,20. ])#arrive to region,when time is 0,1,2 in 1 and 2 region respectively between 1 b 2 region
l=np.array([10.,50.,20.])#leave region, time is 0,1,2 from 1 and 2 region respectively between 1 b 2


u0=100.#initial value of population in 1 and 2 region respectively,third parameter is constant population abroad 
u1= (u0+a[0]-l[0])#numer of population, time 1
u2= (u1+a[1]-l[1])#numer of population, time 2
u3= (u2+a[2]-l[2])#numer of population, time 3

aa1=((a[0]/u0)+(a[1]/u1)+(a[2]/u2))/3.0#average number of population, who arrives to region 1
#aa2=((a[1]/u0[1])+(a[3]/u1[1])+(a[5]/u2[1]))/3.0#average number of population, who arrives to region 2
print('Average number of population, who arrives to region 1 is', aa1)
al1=((l[0]/u0)+(l[1]/u1)+(l[2]/u2))/3.0#average number of population, who leaves region 1
#al2=((l[1]/u0[1])+(l[3]/u1[1])+(l[5]/u2[1]))/3.0#average number of population, who leaves region 2
print('Average number of population, who leaves region 1 is', al1)
s0= np.array([aa1,al1])#vector of initial parameters
dq=0.00001
funclist=[ ]
t = np.linspace(0.,3.)
delta1=1#weight coefficient for function, which must be minimized


class IVPOPTIM(object):  #Problems that are solved by the Runge-Kutta method
     def __init__(self, f=None, u0=100., t0=0., T=3.0, dt0=0.01, exact=None, desc='', name=''):
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
        ivpoptim.u0=u0#initial parameters
        ivpoptim.T=3.0#Final time of integration
        ivpoptim.rhs =lambda t,u: (s0[0]*u-s0[1]*u)  #right side of ODE system
        ivpoptim.dt0 = 0.01#time step
       
      elif testkey=='2OD1':# a system of differential equations with the first parameter changed by an amount dq
        ivpoptim.u0=u0#initial parameters
        ivpoptim.T=3.0#Final time of integration
        ivpoptim.rhs =lambda t,u: ((s0[0]+dq)*u-s0[1]*u)
        ivpoptim.dt0 = 0.01
      elif testkey=='2OD2':#a system of differential equations with the second parameter changed by an amount dq
         ivpoptim.u0=u0#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u:(s0[0]*u-(s0[1]+dq)*u)
         ivpoptim.dt0 = 0.01
                    
      elif testkey=='2OD11':#a system of differential equations with the first parameter changed by an amount 2dq
         ivpoptim.u0=u0#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u: ((s0[0]+2*dq)*u-s0[1]*u)
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD12':#a system of differential equations with the first and second parameter changed by an amount dq
         ivpoptim.u0=u0#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u:((s0[0]+dq)*u-(s0[1]+dq)*u)
                   
      elif testkey=='2OD22':#a system of differential equations with the second parameter changed by an amount 2dq
         ivpoptim.u0=u0#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs =lambda t,u:(s0[0]*u-(s0[1]+2*dq)*u)
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
t4=np.array([t[1667], t[3334], t[5001]])
ivp=detest('2OD1')
myivp1=ivp#load initial parameters from problem
t,u = rk44(myivp1)#approximate soluton of problem  
u5=np.array([u[1667], u[3334], u[5001]])

ivp=detest('2OD2')
myivp2=ivp #load initial parameters from problem
t,u = rk44(myivp2)#approximate soluton of problem  
u6=np.array([u[1667], u[3334], u[5001]])

x0=np.array([u4[0],u4[1],u4[2]])#array of solution in moment of time 1,2,3
xfact=np.array([u1,u2,u3])#fact values in moment of time 1,2,3
x1=np.array([u5[0],u5[1],u5[2]])#solutions with with the first parameter changed by an amount dq 
x2=np.array([u6[0],u6[1],u6[2]])#solutions with with the second parameter changed by an amount dq

func=delta1*sum((xfact[0:3]-x0[0:3])**2)#function which must be minimized
funclist.append(func)#value of a function with initial parameters
i=0 #element number in the array of value of a functions           
#print(func)                  
v1=np.array([((x1[0]-x0[0])/dq), ((x1[1]-x0[1])/dq),((x1[2]-x0[2])/dq)])#derivative with the first parameter
v2=np.array([((x2[0]-x0[0])/dq), ((x2[1]-x0[1])/dq),((x2[2]-x0[2])/dq)])#derivative with the second parameter

derv1 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v1[0:3]))#derivative of function with the first parameter
derv2 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v2[0:3]))#derivative of function with the second parameter

gradv=np.array([derv1,derv2])#gradient
norma=ln.norm(gradv)#norm of gradient

if (norma<=eps1):
        s0=s0  
                           
while(norma>=eps1):
  
    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD11')
    myivp6=ivp#load initial parameters from problem
    t,u = rk44(myivp6)#approximate soluton of problem    
    u7=np.array([u[1667], u[3334], u[5001]])
         
    ivp=detest('2OD12')
    myivp7=ivp#load initial parameters from problem
    t,u = rk44(myivp7)#approximate soluton of problem     
    u8=np.array([u[1667], u[3334], u[5001]])   
         
    ivp=detest('2OD22')
    myivp10=ivp#load initial parameters from problem
    t,u = rk44(myivp10)#approximate soluton of problem    
    u9=np.array([u[1667], u[3334], u[5001]])
                                    
    x11=np.array([u7[0],u7[1],u7[2]])#solutions with with the first parameter changed by an amount 2dq
    x12 = np.array([u8[0],u8[1],u8[2]])#solutions with with the first and second parameter changed by an amount dq 
    x22= np.array([u9[0],u9[1],u9[2]])#solutions with with the second parameter changed by an amount 2dq
    
       
    v11=np.array([((x11[0]-(2*x1[0]-x0[0]))/((dq)**2)), ((x11[1]-(2*x1[1]-x0[1]))/((dq)**2)),((x11[2]-(2*x1[2]-x0[2]))/((dq)**2))])#second derivative with 1 parameter of function
         
    v21=np.array([((x12[0]-x1[0]-x2[0]+x0[0])/((dq)**2)), ((x12[1]-x1[1]-x2[1]+x0[1])/((dq)**2)),((x12[2]-x1[2]-x2[2]+x0[2])/((dq)**2))])#derivative with 1 and 2 parameter of function 
            
    v22=np.array([((x22[0]-(2*x2[0]-x0[0]))/((dq)**2)), ((x22[1]-(2*x2[1]-x0[1]))/((dq)**2)),((x22[2]-(2*x2[2]-x0[2]))/((dq)**2))])#second derivative with 2 parameter of function
         
    
    #Construct the Hessian matrix:
    derv11=delta1*sum(2*(v1[0:3])**2-2*(xfact[0:3]-x0[0:3])*v11[0:3])
    derv12 =delta1*sum(2*(v1[0:3])*v2[0:3]-2*(xfact[0:3]-x0[0:3])*v21[0:3])
        
    derv21 =delta1*sum(2*(v1[0:3])*v2[0:3]-2*(xfact[0:3]-x0[0:3])*v21[0:3])
    derv22=delta1*sum(2*(v2[0:3])**2-2*(xfact[0:3]-x0[0:3])*v22[0:3])
    
    hesse=np.array([[derv11,derv12],[derv21,derv22]])#The Matrix of Hesse
    #print(hesse)
    E=np.diag((1.,1.))#identity matrix
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
    x0=np.array([u4[0],u4[1],u4[2]])#solutions with new parameters in time 1,2,3        
    func1=delta1*sum((xfact[0:3]-x0[0:3])**2)#Find value of function with new parameters    
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
                                                  
                x1=np.array([u5[0],u5[1],u5[2]])#solutions with with the first parameter changed by an amount dq
                x2=np.array([u6[0][0],u6[1][0],u6[2][0]])#solutions with with the second parameter changed by an amount dq
                
                v1=np.array([((x1[0]-x0[0])/dq), ((x1[1]-x0[1])/dq),((x1[2]-x0[2])/dq)])#derivative with the first parameter
                v2=np.array([((x2[0]-x0[0])/dq), ((x2[1]-x0[1])/dq),((x2[2]-x0[2])/dq)])#derivative with the second parameter
                
                derv1 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v1[0:3]))#derivative of function with the first parameter
                derv2 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v2[0:3]))#derivative of function with the second parameter)
                
                gradv=np.array([derv1,derv2])#gradient                
                norma=ln.norm(gradv)#norm of gradient
                s0=s1   
    if (funclist[i-1]<=funclist[i]):
              mu=(mu0)*2#double the step
              mu0=mu   
              s0=s1+p.T#take the previous values of the parameters and return to the beginning of the cycle to recalculate the Hesse matrix   
              
    print(mu0)
    print(func1)
    print(norma)              
    print(s0)           
  
                 
             
