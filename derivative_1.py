from scipy.misc import derivative
from sympy import *
import numpy
import math
from sympy import symbols, diff
x, y, z,k,l,m = symbols('x y z k l m', real=True)
#1. Derivative for for function with 1 variable

def f1(x):   #function 
    return 5*x**2+1

def d_f1(x):  #derivative  
    return f1(x).diff(x)
print('derivative for function with 1 variable:', d_f1(x))

def d_f1_v(x):#value of derivative in point
    
    h=10**(-6)
    return (f1(x+h)-f1(x))/h
print('value of derivative in point x=5:', d_f1_v(5))


#2. Derivative for for function with 2 variable


def f2(x,y):   #function 
    return 5*y*x**2+x*y**3

def d_f2_dx(x,y):  #derivative,x  
    return f2(x,y).diff(x)

print('derivative for function with 2 variable,x:', d_f2_dx(x,y))

def d_f2_dx_v(x,y):#value of derivative in point
    
    h=10**(-6)
    return (f2(x+h,y)-f2(x,y))/h
print('value of derivative in point x=5, y=2:', d_f2_dx_v(5,2))

def d_f2_dy(x,y):  #derivative,y  
    return f2(x,y).diff(y)


print('derivative for function with 2 variable,y:', d_f2_dx(x,y))


def d_f2_dy_v(x,y):#value of derivative in point
    
    h=10**(-6)
    return (f2(x,y+h)-f2(x,y))/h
print('value of derivative in point x=5, y=2:', d_f2_dy_v(5,2))



#3. Derivative for for function with 6 variable


def f3(x,y,z,k,l,m):   #function 
    return 7*x*y + x*z *k**2+ (x**3)*l*(m**4) + z*y*m**3

def d_f3_dx(x,y,z,k,l,m ):  #derivative,x  
    return f3(x,y,z,k,l,m ).diff(x)

print('derivative for function with 6 variable,x:', d_f3_dx(x,y,z,k,l,m ))

def d_f3_dx_v(x,y,z,k,l,m):#value of derivative in point
    
    h=10**(-6)
    return (f3(x+h,y,z,k,l,m )-f3(x,y,z,k,l,m ))/h
print('value of derivative in point x=1, y=1, z=1,k=1,l=1,m=1:', d_f3_dx_v(1,1,1,1,1,1))

def d_f3_dy(x,y,z,k,l,m ):  #derivative,y 
    return f3(x,y,z,k,l,m ).diff(y)

print('derivative for function with 6 variable,y:', d_f3_dy(x,y,z,k,l,m ))

def d_f3_dy_v(x,y,z,k,l,m):#value of derivative in point

    
    h=10**(-6)
    return (f3(x,y+h,z,k,l,m )-f3(x,y,z,k,l,m ))/h
print('value of derivative in point x=1, y=1, z=1,k=1,l=1,m=1:', d_f3_dy_v(1,1,1,1,1,1))



def d_f3_dz(x,y,z,k,l,m ):  #derivative,z 
    return f3(x,y,z,k,l,m ).diff(z)

print('derivative for function with 6 variable,z:', d_f3_dz(x,y,z,k,l,m ))

def d_f3_dz_v(x,y,z,k,l,m):#value of derivative in point
    
    h=10**(-6)
    return (f3(x,y,z+h,k,l,m )-f3(x,y,z,k,l,m ))/h
print('value of derivative in point x=1, y=1, z=1,k=1,l=1,m=1:', d_f3_dz_v(1,1,1,1,1,1))


def d_f3_dk(x,y,z,k,l,m ):  #derivative,k 
    return f3(x,y,z,k,l,m ).diff(k)

print('derivative for function with 6 variable,k:', d_f3_dk(x,y,z,k,l,m ))

def d_f3_dk_v(x,y,z,k,l,m):#value of derivative in point
    
    h=10**(-6)
    return (f3(x,y,z,k+h,l,m )-f3(x,y,z,k,l,m ))/h
print('value of derivative in point x=1, y=1, z=1,k=1,l=1,m=1:', d_f3_dk_v(1,1,1,1,1,1))


def d_f3_dl(x,y,z,k,l,m ):  #derivative,x  
    return f3(x,y,z,k,l,m ).diff(l)

print('derivative for function with 6 variable,l:', d_f3_dl(x,y,z,k,l,m ))

def d_f3_dl_v(x,y,z,k,l,m):#value of derivative in point
    
    h=10**(-6)
    return (f3(x,y,z,k,l+h,m )-f3(x,y,z,k,l,m ))/h
print('value of derivative in point x=1, y=1, z=1,k=1,l=1,m=1:', d_f3_dl_v(1,1,1,1,1,1))


def d_f3_dm(x,y,z,k,l,m ):  #derivative,x  
    return f3(x,y,z,k,l,m ).diff(m)

print('derivative for function with 6 variable,m:', d_f3_dm(x,y,z,k,l,m ))

def d_f3_dm_v(x,y,z,k,l,m):#value of derivative in point
    
    h=10**(-6)
    return (f3(x,y,z,k,l,m+h )-f3(x,y,z,k,l,m ))/h
print('value of derivative in point x=1, y=1, z=1,k=1,l=1,m=1:', d_f3_dm_v(1,1,1,1,1,1))

