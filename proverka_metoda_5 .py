from decimal import Decimal 
from math import exp
import numpy as np
from numpy import *
import numpy.linalg as ln
from matplotlib import mlab
import matplotlib.pyplot as plt
import pylab
import nodepy
from nodepy import rk

eps1 = 0.0001#accuracy
mu0=10**2 #value for changing the Hessian matrix
a0=3#initial value of parameter

u0=1.#initial solution in moment time 0 
dq=0.00001
funclist=[ ]
delta1=1#weight coefficient for function, which must be minimized


class IVPOPTIM(object):  #Problems that are solved by the Runge-Kutta method
     def __init__(self, f=None, u0=10., t0=0., T=3.0, dt0=0.01, exact=None, desc='', name=''):
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
      if testkey=='Differential equation for exact solution':#differential equation
                         #with parameter which is (-1), i.e. we think, that a0=-1,
                         # it is necessary to find "experimental" solution which is identically to exact solution
        ivpoptim.u0=u0#initial value
        ivpoptim.T=3.0#Final time of integration        
        ivpoptim.rhs =lambda t,z:-1*z  #right side of ODE 
        ivpoptim.dt0 = 0.01#time step
        
      else: raise Exception('Unknown Detest problem')
      ivpoptim.name=testkey
      ivpoptim.description='Problem '+testkey+' of the non-stiff DETEST suite.'
      return ivpoptim  

rk44 = rk.loadRKM('RK44')#load runge-kutta
ivp=detest('Differential equation for exact solution')
myivp=ivp#load initial parameters from problem
t,z = rk44(myivp)#approximate soluton of problem
r=abs(t[1]-t[0])#the size of step
r1=int(round((1/r)+1))#sequence number for the first point of time
r2=int(round((2/r)+1))#sequence number for the second point of time
r3=int(round((3/r)+1))#sequence number for the third point of time
z1=np.array([z[r1], z[r2], z[r3]])#"experimental" solutions in moment of time 1,2,3
def detest(testkey):#Description of problem
      import numpy as np
      ivpoptim=IVPOPTIM()
      if testkey=='Differential equation':#differential equation with parameter a0 which is not (-1)      
        #print('\nDifferential equation -----') 
        ivpoptim.u0=u0#initial value
        ivpoptim.T=3.0#Final time of integration
        ivpoptim.rhs =lambda t,u:a0*u  #right side of ODE 
        ivpoptim.dt0 = 0.01#time step
       
      elif testkey=='Differential equation1':# differential equation with parameter changed by an amount dq
        #print('\n1 Differential equation 1 -----')

        ivpoptim.u0=u0#initial value
        ivpoptim.T=3.0#Final time of integration
        ivpoptim.rhs =lambda t,u: (a0+dq)*u
        ivpoptim.dt0 = 0.01
      
                    
      elif testkey=='Differential equation11':#differential equation with parameter changed by an amount 2dq
         #print('\n2 Differential equation 2 -----')

         ivpoptim.u0=u0#initial value
         ivpoptim.T=3.0#Final time of integration
         ivpoptim.rhs = lambda t,u: (a0+2*dq)*u
         ivpoptim.dt0 = 0.01     
       
      else: raise Exception('Unknown Detest problem')
      ivpoptim.name=testkey
      ivpoptim.description='Problem '+testkey+' of the non-stiff DETEST suite.'
      return ivpoptim  
  

rk44 = rk.loadRKM('RK44')#load runge-kutta
ivp=detest('Differential equation')
myivp1=ivp#load initial parameters from problem
t,u = rk44(myivp1)#approximate soluton of problem   
u1=np.array([u[r1], u[r2], u[r3]])#approximate soluton in moment of time 1,2,3

ivp=detest('Differential equation1')
myivp2=ivp#load initial parameters from problem
t,u = rk44(myivp2)#approximate soluton of problem  
u2=np.array([u[r1], u[r2], u[r3]])



x0=np.array([u1[0],u1[1],u1[2]])#array of solution in moment of time 1,2,3
xfact=np.array([z1[0],z1[1],z1[2]])#fact values in moment of time 1,2,3
x1=np.array([u2[0],u2[1],u2[2]])#solutions with with the first parameter changed by an amount dq 

#fig=[plt.plot(t, z,"r-")]
func=delta1*sum((xfact[0:3]-x0[0:3])**2) #function which must be minimized
funclist.append(func)#value of a function with initial parameters
i=0 #element number in the array of value of a functions
pogr1=abs(z1[0]-u1[0])#error in 1 point
pogr2=abs(z1[1]-u1[1])#error in 2 point
pogr3=abs(z1[2]-u1[2])#error in 3 point
vectpogr=[pogr1,pogr2,pogr3]
totalpogr=ln.norm(vectpogr)# total error
print('error in 1 time:')
print(pogr1)
    
print('error in 2 time:')
print(pogr2)
    
print('error in 3 time:')
print(pogr3)
    
print('total error:')
print(totalpogr)
print('number of iteration:')
print(i)
print('value for changing the Hessian matrix in 0 step:')
print(mu0)
print('value of function which must be minimized in 0 step:')
print(func)                  
v1=np.array([((x1[0]-x0[0])/dq), ((x1[1]-x0[1])/dq),((x1[2]-x0[2])/dq)])#derivative with the parameter
derv1 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v1[0:3]))#derivative of function with the first parameter

print(derv1)
gradv=derv1#gradient
print('gradient in 0 step:')
print(gradv)
norma=ln.norm(gradv)#norm of gradient
print('norm of gradient in 0 step:')
print(norma)

ilist=[i]#list of index
alist=[a0]#list of parameters a0
totalpogrlist=[totalpogr]#list of total error
while(norma>=eps1):
    
    
    rk44 = rk.loadRKM('RK44')#load runge-kutta
    ivp=detest('Differential equation11')
    myivp3=ivp#load initial parameters from problem
    t,u = rk44(myivp3)#approximate soluton of problem    

    u3=np.array([u[r1], u[r2], u[r3]])        
                                        
    x11=np.array([u3[0],u3[1],u3[2]])#solutions with with the  parameter changed by an amount 2dq
           
    v11=np.array([((x11[0]-(2*x1[0]-x0[0]))/((dq)**2)), ((x11[1]-(2*x1[1]-x0[1]))/((dq)**2)),((x11[2]-(2*x1[2]-x0[2]))/((dq)**2))])#second derivative with parameter of function
         
   #Construct the Hessian matrix:
    derv11=delta1*sum(2*(v1[0:3])**2-2*(xfact[0:3]-x0[0:3])*v11[0:3])
     
    hesse=derv11#The Matrix of Hesse
    
    E=1#identity matrix
    chhesse=hesse+mu0*E#modified Hessian matrix
    inversechhesse=1/(chhesse)# inverse matrix of modified Hessian matrix    
    p=np.dot(inversechhesse,(gradv.T))#vector for calculating  new parameter   
    a1=a0-p.T#new parameter
    a0=a1
        
    rk44 = rk.loadRKM('RK44')#load runge-kutta, find solution with new parameter
    ivp=detest('Differential equation')
    myivp1=ivp#load initial value from problem
    t,u = rk44(myivp1)#approximate soluton of problem      
    u1=np.array([u[r1], u[r2], u[r3]])
    #fig.append(plt.plot(t, u,"y--",lw=0.5))
    pogr1=abs(z1[0]-u1[0])
    pogr2=abs(z1[1]-u1[1])
    pogr3=abs(z1[2]-u1[2])
    vectpogr=[pogr1,pogr2,pogr3]
    totalpogr=ln.norm(vectpogr)
    x0=np.array([u1[0],u1[1],u1[2]])#solutions with new parameter in time 1,2,3        
    func1=delta1*sum((xfact[0:3]-x0[0:3])**2) #Find value of function with new parameter    
    funclist.append(func1)#value of a function with initial parameters       
    i=i+1 #element number in the array of value of a functions with new parameters
                    
    #compare the new and previous values of the function:          
    if (funclist[i-1]> funclist[i]):
               
                a0=a1#take new values of the parameters
                mu=(mu0)/2#halve the step
                mu0=mu
                #Find gradien and norm of gradient and return to the beginning of the cycle to calculate the Hesse matrix      
                rk44 = rk.loadRKM('RK44')#load runge-kutta
                ivp=detest('Differential equation1')
                myivp2=ivp#load initial parameters from problem
                t,u = rk44(myivp2)#approximate soluton of problem             
                u2=np.array([u[r1], u[r2], u[r3]])                    
                                             
                x1=np.array([u2[0],u2[1],u2[2]])#solutions with with the parameter changed by an amount dq                
                
                v1=np.array([((x1[0]-x0[0])/dq), ((x1[1]-x0[1])/dq),((x1[2]-x0[2])/dq)])#derivative with the first parameter                
                
                derv1 = -2*delta1*sum((xfact[0:3]-x0[0:3])*(v1[0:3]))#derivative of function with the first parameter
                                
                gradv=derv1#gradient                
                norma=ln.norm(gradv)   #norm of gradient
                a0=a1
                alist.append(a0)#list of parameteres a0
                
    if (funclist[i-1]<=funclist[i]):
              mu=(mu0)*2#double the step
              mu0=mu
              a0=a0+p.T#take the previous values of the parameters and return to the beginning of the cycle to recalculate the Hesse matrix   
              
    
    #fig.append(plt.plot(t, u,"g-",lw=(2+0.5*i)/(i+0.2)))
    print('number of iteration:')
    print(i)
    print('value for changing the Hessian matrix or step:')
    print(mu0)
    print('value of function which must be minimized:')
    print(func1)
    print('norm of gradient:')
    print(norma)
    print('results of parameters identification:')
    print(a0)
    ilist.append(i)
    totalpogrlist.append(totalpogr)
    print('error in 1 time:')
    print(pogr1)
    
    print('error in 2 time:')
    print(pogr2)
    
    print('error in 3 time:')
    print(pogr3)
    
    print('total error:')
    print(totalpogr)
    
maxpogr=max(totalpogrlist)#maximum total error    
print('all number of iteration:')
print(i)
#jlist=[]#empty list of index for line thickness
#j=0
#plot for solution
print(ilist)
print(alist)

plt.axis([0, 3,0,2])
plt.plot(t, z,"k-",lw=0.8)
fig=[]
#plot of "exact" solution
for a0 in alist:#solution of ODE1 on i step with new parameters a0
     #j=j+1
     #jlist.appe
     ivp=detest('Differential equation')
     myivp1=ivp#load initial parameters from problem
     t,u = rk44(myivp1)#approximate soluton of problem
     
     
     plt.plot(t, u,"y--",lw=0.5)
     fig.append(plt.plot(t, u,"y--",lw=0.5)) 
    
plt.plot(t, z,"k-",lw=0.8)
plt.xlabel('t')
plt.ylabel('u')
plt.legend(['z(t),dz/dt=-1*z','u(t), du/dt=a0*z'], loc = 'right up')              
plt.grid(fig)
plt.show(fig)

#plot for total error in each step
fig1=[]
plt.axis([0,ilist[-1]+1,0,maxpogr+1])
for i in ilist:
   plt.plot(i, totalpogrlist[i],"ko",lw=0.09)
   fig1.append(plt.plot(i, totalpogrlist[i],"ko",lw=0.09))



plt.xlabel('i')
plt.ylabel('totalpogr')
plt.legend(['total error'], loc = 'right up')              
plt.grid(fig1)
plt.show(fig1)

