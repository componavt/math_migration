from nodepy import ivp
import nodepy.linear_multistep_method as lm
myivp = ivp.detest('B1')#loading the initial data of task B1, which is described in ivp.py
print(myivp)
ab4=lm.Adams_Bashforth(4)#call method Adams-Bashforth
t, y = ab4(myivp)# integrate of B1 by method Adams-Bashforth
print(t,y)
