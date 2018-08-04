 
from math import exp
import numpy as np
from numpy import *
import numpy.linalg as ln
import nodepy
from nodepy import rk
eps1 = 0.0001#accuracy
mu0=10**4 #value for changing the Hessian matrix
a = np.array([20.,10.,10.,0.,20.,0. ])#arrive to region,when time is 0,1,2 in 1 and 2 region respectively between 1 b 2 region
l=np.array([10.,30.,50.,40.,20.,0.])#leave region, time is 0,1,2 from 1 and 2 region respectively between 1 b 2


u0=np.array([100.,70.])#initial value of population in 1 and 2 region respectively,third parameter is constant population abroad 
u1= np.array([(u0[0]+a[0]-l[0]),(u0[1]+a[1]-l[1])])#numer of population, time 1
u2= np.array([(u1[0]+a[2]-l[2]),(u1[1]+a[3]-l[3])])#numer of population, time 2
u3= np.array([(u2[0]+a[4]-l[4]),(u2[1]+a[5]-l[5])])#numer of population, time 3

aa1=((a[0]/u0[0])+(a[2]/u1[0])+(a[4]/u2[0]))/3.0#average number of population, who arrives to region 1
aa2=((a[1]/u0[1])+(a[3]/u1[1])+(a[5]/u2[1]))/3.0#average number of population, who arrives to region 2

al1=((l[0]/u0[0])+(l[2]/u1[0])+(l[4]/u2[0]))/3.0#average number of population, who leaves region 1
al2=((l[1]/u0[1])+(l[3]/u1[1])+(l[5]/u2[1]))/3.0#average number of population, who leaves region 2

s0= np.array([aa1,aa2,al1,al2])#vector of initial parameters
dq=0.00001
funclist=[ ]
t = np.linspace(0.,3.)
delta1=0.5#weight coefficient for function, which must be minimized,delta1+delta2=1
delta2=0.5#weight coefficient for function, which must be minimized,delta1+delta2=1

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
        ivpoptim.rhs =lambda t,u: np.array([(s0[0]*u[0]-s0[2]*u[0]),(s0[1]*u[1]-s0[3]*u[1])])  #right side of ODE system
        ivpoptim.dt0 = 0.01#time step
       
      elif testkey=='2OD1':# a system of differential equations with the first parameter changed by an amount dq
        ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
        ivpoptim.T=3.0#Final time of integration
        ivpoptim.rhs =lambda t,u: np.array([((s0[0]+dq)*u[0]-s0[2]*u[0]),(s0[1]*u[1]-s0[3]*u[1])])
        ivpoptim.dt0 = 0.01
      elif testkey=='2OD2':#a system of differential equations with the second parameter changed by an amount dq
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u:np.array([(s0[0]*u[0]-s0[2]*u[0]),((s0[1]+dq)*u[1]-s0[3]*u[1])])
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD3':#a system of differential equations with the third parameter changed by an amount dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u:np.array([(s0[0]*u[0]-(s0[2]+dq)*u[0]),(s0[2]*u[1]-s0[3]*u[1])]) 
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD4':#a system of differential equations with the third parameter changed by an amount dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u:np.array([(s0[0]*u[0]-s0[2]*u[0]),(s0[1]*u[1]-(s0[3]+dq)*u[1])]) 
         ivpoptim.dt0 = 0.01  
         
      elif testkey=='2OD11':#a system of differential equations with the first parameter changed by an amount 2dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u: np.array([((s0[0]+2*dq)*u[0]-s0[2]*u[0]),(s0[1]*u[1]-s0[3]*u[1])])
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD12':#a system of differential equations with the first and second parameter changed by an amount dq
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u:np.array([((s0[0]+dq)*u[0]-s0[1]*u[0]),((s0[2]+dq)*u[1]-s0[3]*u[1])])
      elif testkey=='2OD13':#a system of differential equations with the first and third parameter changed by an amount dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u: np.array([((s0[0]+dq)*u[0]-(s0[2]+dq)*u[0]),(s0[1]*u[1]-s0[3]*u[1])])         
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD14':#a system of differential equations with the first and fourth parameter changed by an amount dq
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u: np.array([((s0[0]+dq)*u[0]-s0[2]*u[0]),(s0[1]*u[1]-(s0[3]+dq)*u[1])])         
         ivpoptim.dt0 = 0.01
         
      elif testkey=='2OD22':#a system of differential equations with the second parameter changed by an amount 2dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs =lambda t,u:np.array([(s0[0]*u[0]-s0[2]*u[0]),((s0[1]+2*dq)*u[1]-s0[3]*u[1])])
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD23':#a system of differential equations with the second and third parameter changed by an amount dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs =lambda t,u:np.array([(s0[0]*u[0]-(s0[2]+dq)*u[0]),((s0[1]+dq)*u[1]-s0[3]*u[1])]) 
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD24':#a system of differential equations with the second and fourth parameter changed by an amount dq

         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs =lambda t,u:np.array([(s0[0]*u[0]-s0[2]*u[0]),((s0[1]+dq)*u[1]-(s0[3]+dq)*u[1])]) 
         ivpoptim.dt0 = 0.01   
      
      elif testkey=='2OD33':#a system of differential equations with the third parameter changed by an amount 2dq
 
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u:np.array([(s0[0]*u[0]-(s0[2]+2*dq)*u[0]),(s0[1]*u[1]-s0[3]*u[1])])
         ivpoptim.dt0 = 0.01
      elif testkey=='2OD34':#a system of differential equations with the third and fourth parameter changed by an amount 2dq
 
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u:np.array([(s0[0]*u[0]-(s0[2]+dq)*u[0]),(s0[1]*u[1]-(s0[3]+dq)*u[1])])
         ivpoptim.dt0 = 0.01
         
      elif testkey=='2OD44':#a system of differential equations with the fourth parameter changed by an amount 2dq
 
         ivpoptim.u0=np.array([u0[0],u0[1]])#initial parameters
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u:np.array([(s0[0]*u[0]-s0[2]*u[0]),(s0[1]*u[1]-(s0[3]+2*dq)*u[1])])
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

ivp=detest('2OD4')
myivp5=ivp#load initial parameters from problem
t,u = rk44(myivp5)#approximate soluton of problem
u8=np.array([u[1667], u[3334], u[5001]])

x0=np.array([u4[0][0],u4[1][0],u4[2][0],u4[0][1],u4[1][1],u4[2][1]])#array of solution in moment of time 1,2,3
xfact=np.array([u1[0],u2[0],u3[0],u1[1],u2[1],u3[1]])#fact values in moment of time 1,2,3
x1=np.array([u5[0][0],u5[1][0],u5[2][0],u5[0][1],u5[1][1],u5[2][1]])#solutions with with the first parameter changed by an amount dq 
x2=np.array([u6[0][0],u6[1][0],u6[2][0],u6[0][1],u6[1][1],u6[2][1]])#solutions with with the second parameter changed by an amount dq
x3=np.array([u7[0][0],u7[1][0],u7[2][0],u7[0][1],u7[1][1],u7[2][1]])#solutions with with the third parameter changed by an amount dq
x4=np.array([u8[0][0],u8[1][0],u8[2][0],u8[0][1],u8[1][1],u8[2][1]])#solutions with with the fourth parameter changed by an amount dq
func=delta1*sum((xfact[0:3]-x0[0:3])**2)+delta2*sum((xfact[3:]-x0[3:])**2)#function which must be minimized
funclist.append(func)#value of a function with initial parameters
i=0 #element number in the array of value of a functions           
#print(func)                  
v1=np.array([((x1[0]-x0[0])/dq), ((x1[1]-x0[1])/dq),((x1[2]-x0[2])/dq), ((x1[3]-x0[3])/dq),((x1[4]-x0[4])/dq), ((x1[5]-x0[5])/dq)])#derivative with the first parameter
v2=np.array([((x2[0]-x0[0])/dq), ((x2[1]-x0[1])/dq),((x2[2]-x0[2])/dq), ((x2[3]-x0[3])/dq),((x2[4]-x0[4])/dq), ((x2[5]-x0[5])/dq)])#derivative with the second parameter
v3=np.array([((x3[0]-x0[0])/dq), ((x3[1]-x0[1])/dq),((x3[2]-x0[2])/dq), ((x3[3]-x0[3])/dq),((x3[4]-x0[4])/dq), ((x3[5]-x0[5])/dq)])#derivative with the third parameter
v4=np.array([((x4[0]-x0[0])/dq), ((x4[1]-x0[1])/dq),((x4[2]-x0[2])/dq), ((x4[3]-x0[3])/dq),((x4[4]-x0[4])/dq), ((x4[5]-x0[5])/dq)])#derivative with the fourth parameter

derv1 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v1[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v1[3:]))#derivative of function with the first parameter
derv2 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v2[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v2[3:]))#derivative of function with the second parameter
derv3 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v3[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v3[3:]))#derivative of function with the third parameter
derv4 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v4[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v4[3:]))#derivative of function with the fourth parameter
gradv=np.array([derv1,derv2,derv3,derv4])#gradient
norma=ln.norm(gradv)#norm of gradient

if (norma<=eps1):
        s0=s0  
                           
while(norma>=eps1):
  
    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD11')
    myivp6=ivp#load initial parameters from problem
    t,u = rk44(myivp6)#approximate soluton of problem    
    u9=np.array([u[1667], u[3334], u[5001]])

    
    ivp=detest('2OD12')
    myivp7=ivp#load initial parameters from problem
    t,u = rk44(myivp7)#approximate soluton of problem     
    u10=np.array([u[1667], u[3334], u[5001]])
    
   
    ivp=detest('2OD13')
    myivp8=ivp#load initial parameters from problem
    t,u = rk44(myivp8)#approximate soluton of problem     
    u11=np.array([u[1667], u[3334], u[5001]])
    
    ivp=detest('2OD14')
    myivp9=ivp#load initial parameters from problem
    t,u = rk44(myivp9)#approximate soluton of problem     
    u12=np.array([u[1667], u[3334], u[5001]])
    
    ivp=detest('2OD22')
    myivp10=ivp#load initial parameters from problem
    t,u = rk44(myivp10)#approximate soluton of problem    
    u13=np.array([u[1667], u[3334], u[5001]])
    
    ivp=detest('2OD23')
    myivp11=ivp#load initial parameters from problem
    t,u = rk44(myivp11)#approximate soluton of problem     
    u14=np.array([u[1667], u[3334], u[5001]])

    ivp=detest('2OD24')
    myivp12=ivp#load initial parameters from problem
    t,u = rk44(myivp12)#approximate soluton of problem     
    u15=np.array([u[1667], u[3334], u[5001]])
    
    #rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('2OD33')
    myivp13=ivp#load initial parameters from problem
    t,u = rk44(myivp13)#approximate soluton of problem       
    u16=np.array([u[1667], u[3334], u[5001]])

    ivp=detest('2OD34')
    myivp14=ivp#load initial parameters from problem
    t,u = rk44(myivp14)#approximate soluton of problem       
    u17=np.array([u[1667], u[3334], u[5001]])
    ivp=detest('2OD44')
    myivp15=ivp#load initial parameters from problem
    t,u = rk44(myivp15)#approximate soluton of problem       
    u18=np.array([u[1667], u[3334], u[5001]])
                              
    x11=np.array([u9[0][0],u9[1][0],u9[2][0],u9[0][1],u9[1][1],u9[2][1]])#solutions with with the first parameter changed by an amount 2dq
    x12 = np.array([u10[0][0],u10[1][0],u10[2][0],u10[0][1],u10[1][1],u10[2][1]])#solutions with with the first and second parameter changed by an amount dq 
    
    x13= np.array([u11[0][0],u11[1][0],u11[2][0],u11[0][1],u11[1][1],u11[2][1]])#solutions with with the first and third parameter changed by an amount dq 
    x14= np.array([u12[0][0],u12[1][0],u12[2][0],u12[0][1],u12[1][1],u12[2][1]])#solutions with with the firdst and fourth and third parameter changed by an amount dq

    x22= np.array([u13[0][0],u13[1][0],u13[2][0],u13[0][1],u13[1][1],u13[2][1]])#solutions with with the second parameter changed by an amount 2dq
    
    x23= np.array([u14[0][0],u14[1][0],u14[2][0],u14[0][1],u14[1][1],u14[2][1]])#solutions with with the second and third parameter changed by an amount dq
    x24= np.array([u15[0][0],u15[1][0],u15[2][0],u15[0][1],u15[1][1],u15[2][1]])#solutions with with the second and fourth parameter changed by an amount dq
    x33=np.array([u16[0][0],u16[1][0],u16[2][0],u16[0][1],u16[1][1],u16[2][1]])#solutions with with the third parameter changed by an amount 2dq              
    x34=np.array([u17[0][0],u17[1][0],u17[2][0],u17[0][1],u17[1][1],u17[2][1]])#solutions with with the third and fourth parameter changed by an amount dq
    x44=np.array([u18[0][0],u18[1][0],u18[2][0],u18[0][1],u18[1][1],u18[2][1]])#solutions with with the fourth parameter changed by an amount 2dq 
    
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
    v41=np.array([((x14[0]-x1[0]-x4[0]+x0[0])/((dq)**2)), ((x14[1]-x1[1]-x4[1]+x0[1])/((dq)**2)),((x14[2]-x4[2]-x1[2]+x0[2])/((dq)**2)),
                           ((x14[3]-x1[3]-x4[3]+x0[3])/((dq)**2)),((x14[4]-x1[4]-x4[4]+x0[4])/((dq)**2)), ((x14[5]-x1[5]-x4[5]+x0[5])/((dq)**2))])#derivative with 1 and 4 parameter of function


    v42=np.array([((x24[0]-x4[0]-x2[0]+x0[0])/((dq)**2)), ((x24[1]-x4[1]-x2[1]+x0[1])/((dq)**2)),((x24[2]-x4[2]-x2[2]+x0[2])/((dq)**2)),
                           ((x24[3]-x4[3]-x2[3]+x0[3])/((dq)**2)),((x24[4]-x4[4]-x2[4]+x0[4])/((dq)**2)), ((x24[5]-x4[5]-x2[5]+x0[5])/((dq)**2))])#derivative with 2 and 4 parameter of function
    
    v43=np.array([((x34[0]-x3[0]-x4[0]+x0[0])/((dq)**2)), ((x34[1]-x3[1]-x4[1]+x0[1])/((dq)**2)),((x34[2]-x3[2]-x4[2]+x0[2])/((dq)**2)),
                           ((x34[3]-x3[3]-x4[3]+x0[3])/((dq)**2)),((x34[4]-x3[4]-x4[4]+x0[4])/((dq)**2)), ((x34[5]-x3[5]-x4[5]+x0[5])/((dq)**2))])#derivative with 3 and 4 parameter of function


    v44=np.array([((x44[0]-(2*x4[0]-x0[0]))/((dq)**2)), ((x44[1]-(2*x4[1]-x0[1]))/((dq)**2)),((x44[2]-(2*x4[2]-x0[2]))/((dq)**2)),
                              ((x44[3]-(2*x4[3]-x0[3]))/((dq)**2)),((x44[4]-(2*x4[4]-x0[4]))/((dq)**2)), ((x44[5]-(2*x4[5]-x0[5]))/((dq)**2))])#second derivative with 4 parameter of function

    #Construct the Hessian matrix:
    derv11=delta1*sum(2*(v1[0:3])**2-2*(xfact[0:3]-x0[0:3])*v11[0:3])+delta2*sum(2*(v1[3:])**2-2*(xfact[3:]-x0[3:])*v11[3:])
    derv12 =delta1*sum(2*(v1[0:3])*v2[0:3]-2*(xfact[0:3]-x0[0:3])*v21[0:3])+delta2*sum(2*(v1[3:])*v2[3:]-2*(xfact[3:]-x0[3:])*v21[3:])
    derv13=delta1*sum(2*(v1[0:3])*v3[0:3]-2*(xfact[0:3]-x0[0:3])*v31[0:3])+delta2*sum(2*(v1[3:])*v3[3:]-2*(xfact[3:]-x0[3:])*v31[3:])

    derv14=delta1*sum(2*(v1[0:3])*v4[0:3]-2*(xfact[0:3]-x0[0:3])*v41[0:3])+delta2*sum(2*(v1[3:])*v4[3:]-2*(xfact[3:]-x0[3:])*v41[3:])
    derv21 =delta1*sum(2*(v1[0:3])*v2[0:3]-2*(xfact[0:3]-x0[0:3])*v21[0:3])+delta2*sum(2*(v1[3:])*v2[3:]-2*(xfact[3:]-x0[3:])*v21[3:])
    derv22=delta1*sum(2*(v2[0:3])**2-2*(xfact[0:3]-x0[0:3])*v22[0:3])+delta2*sum(2*(v2[3:])**2-2*(xfact[3:]-x0[3:])*v22[3:])
    derv23=delta1*sum(2*(v2[0:3])*v3[0:3]-2*(xfact[0:3]-x0[0:3])*v23[0:3])+delta2*sum(2*(v2[3:])*v3[3:]-2*(xfact[3:]-x0[3:])*v23[3:])

    derv24=delta1*sum(2*(v2[0:3])*v4[0:3]-2*(xfact[0:3]-x0[0:3])*v42[0:3])+delta2*sum(2*(v2[3:])*v4[3:]-2*(xfact[3:]-x0[3:])*v42[3:])
    derv31 =delta1*sum(2*(v3[0:3])*v1[0:3]-2*(xfact[0:3]-x0[0:3])*v31[0:3])+delta2*sum(2*(v3[3:])*v1[3:]-2*(xfact[3:]-x0[3:])*v31[3:])
    derv32 =delta1*sum(2*(v3[0:3])*v2[0:3]-2*(xfact[0:3]-x0[0:3])*v23[0:3])+delta2*sum(2*(v3[3:])*v2[3:]-2*(xfact[3:]-x0[3:])*v23[3:])
    derv33=delta1*sum(2*(v3[0:3])**2-2*(xfact[0:3]-x0[0:3])*v33[0:3])+delta2*sum(2*(v3[3:])**2-2*(xfact[3:]-x0[3:])*v33[3:])
    derv34 =delta1*sum(2*(v3[0:3])*v4[0:3]-2*(xfact[0:3]-x0[0:3])*v43[0:3])+delta2*sum(2*(v3[3:])*v4[3:]-2*(xfact[3:]-x0[3:])*v43[3:])     
    derv44=delta1*sum(2*(v4[0:3])**2-2*(xfact[0:3]-x0[0:3])*v44[0:3])+delta2*sum(2*(v4[3:])**2-2*(xfact[3:]-x0[3:])*v44[3:])

    hesse=np.array([[derv11,derv12,derv13,derv14],[derv21,derv22,derv23,derv24],[derv31,derv32,derv33,derv34],[derv14,derv24,derv34,derv44]])#The Matrix of Hesse
    #print(hesse)
    E=np.diag((1.,1.,1.,1.))#identity matrix
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

                ivp=detest('2OD4')
                myivp4=ivp#load initial parameters from problem
                t,u = rk44(myivp4)#approximate soluton of problem
                u8=np.array([u[1667], u[3334], u[5001]])
                
                x1=np.array([u5[0][0],u5[1][0],u5[2][0],u5[0][1],u5[1][1],u5[2][1]])#solutions with with the first parameter changed by an amount dq
                x2=np.array([u6[0][0],u6[1][0],u6[2][0],u6[0][1],u6[1][1],u6[2][1]])#solutions with with the second parameter changed by an amount dq
                x3=np.array([u7[0][0],u7[1][0],u7[2][0],u7[0][1],u7[1][1],u7[2][1]])#solutions with with the third parameter changed by an amount dq
                x4=np.array([u8[0][0],u8[1][0],u8[2][0],u8[0][1],u8[1][1],u8[2][1]])

                v1=np.array([((x1[0]-x0[0])/dq), ((x1[1]-x0[1])/dq),((x1[2]-x0[2])/dq), ((x1[3]-x0[3])/dq),((x1[4]-x0[4])/dq), ((x1[5]-x0[5])/dq)])#derivative with the first parameter
                v2=np.array([((x2[0]-x0[0])/dq), ((x2[1]-x0[1])/dq),((x2[2]-x0[2])/dq), ((x2[3]-x0[3])/dq),((x2[4]-x0[4])/dq), ((x2[5]-x0[5])/dq)])#derivative with the second parameter
                v3=np.array([((x3[0]-x0[0])/dq), ((x3[1]-x0[1])/dq),((x3[2]-x0[2])/dq), ((x3[3]-x0[3])/dq),((x3[4]-x0[4])/dq), ((x3[5]-x0[5])/dq)])#derivative with the third parameter
                v4=np.array([((x4[0]-x0[0])/dq), ((x4[1]-x0[1])/dq),((x4[2]-x0[2])/dq), ((x4[3]-x0[3])/dq),((x4[4]-x0[4])/dq), ((x4[5]-x0[5])/dq)])
                derv1 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v1[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v1[3:]))#derivative of function with the first parameter
                derv2 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v2[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v2[3:]))#derivative of function with the second parameter)
                derv3 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v3[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v3[3:]))#derivative of function with the third parameter
                derv4 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v4[0:3]))-2*delta2*sum((xfact[3:]-x0[3:])*(v4[3:]))

                gradv=np.array([derv1,derv2,derv3,derv4])#gradient                
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
  
                 
             
