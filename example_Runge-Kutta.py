from nodepy import ivp,rk
myivp = ivp.detest('B1')#loading the initial data of task B1, which is described in ivp.py
print(myivp)
rk4 = rk.loadRKM('RK44')#call method Runge - Kutta
t,y = rk4(myivp)#integrate of B1 by method Runge-Kutta
print(t,y)
