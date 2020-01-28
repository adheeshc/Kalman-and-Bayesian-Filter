import numpy as np
from scipy.linalg import block_diag
import math

class ConstantVelocityObject(object):
    def __init__(self, x0=0, vel=1., noise_scale=0.06):
        self.x = x0
        self.vel = vel
        self.noise_scale = noise_scale

    def update(self):
        self.vel += np.random.randn() * self.noise_scale
        self.x += self.vel
        return (self.x, self.vel)

def simulate_vel_system(Q, count):
    obj = ConstantVelocityObject(x0=.0, vel=0.5, noise_scale=Q)
    xs, zs = [], []
    for i in range(count):
        x = obj.update()
        z = sense(x)
        xs.append(x)
        zs.append(z)
    return np.array(xs), np.array(zs)


class ConstantAccObject():
    def __init__(self,x0=0,vel=1,acc=0.1,noise_scale=0.1):
        self.x=x0
        self.vel=vel
        self.acc=acc
        self.acc_noise=noise_scale
    
    def update(self):
        self.acc+=np.random.rand()*self.acc_noise
        self.vel+=self.acc
        self.x+=self.vel
        return (self.x,self.vel,self.acc)

def simulate_acc_system(R,Q,count):
    obj=ConstantAccObject(noise_scale=0)
    xs, zs = [], []
    for i in range(count):
        x = obj.update()
        z = sense(x,R)
        xs.append(x)
        zs.append(z)
    return np.array(xs), np.array(zs)

def sense(x, noise_scale=1.):
    return x[0] + np.random.randn()*noise_scale

def generateWhiteNoise(n,dt,process_var):

	if not (n == 2 or n == 3 or n == 4):
		raise ValueError("num_state_var must be between 2 and 4")
	if n==2:
		Q = [[.25*dt**4,.5*dt**3],[.5*dt**3,dt**2]]
	elif n==3:
		Q = [[.25*dt**4, .5*dt**3, .5*dt**2],
		[ .5*dt**3,    dt**2,       dt],
		[ .5*dt**2,       dt,        1]]
	else:
		Q = [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
		[(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
		[(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
		[(dt**3)/6,  (dt**2)/2 ,  dt,        1.]]

	process_var=block_diag(*[Q]*1) * process_var

	return process_var

class Gaussian:
    def __init__(self,mu,var):
        self.mu=mu
        self.var=var

    def mean(self):
        return self.mu

    def variance(self):
        return self.var


def rk4(y, x, dx, f):
    """computes 4th order Runge-Kutta for dy/dx.
    y is the initial value for y
    x is the initial value for x
    dx is the difference in x (e.g. the time step)
    f is a callable function (y, x) that you supply to 
      compute dy/dx for the specified values.
    """
    
    k1 = dx * f(y, x)
    k2 = dx * f(y + 0.5*k1, x + 0.5*dx)
    k3 = dx * f(y + 0.5*k2, x + 0.5*dx)
    k4 = dx * f(y + k3, x + dx)
    
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.

def fx(x,t):
    return fx.vel
    
def fy(y,t):
    return fy.vel - 9.8*t

class BallTrajectory2D():
    def __init__(self,x0,y0,vel,theta,g=9.8,noise=[0,0]):
        self.x=x0
        self.y=y0
        self.t=0
        theta=math.radians(theta)
        fx.vel=math.cos(theta)*vel
        fy.vel=math.sin(theta)*vel
        self.g=g
        self.noise=noise

    def rk4(self,y,x,dx,f): #RUNGE KUTTA 4th ORDER
        k1=dx * f(y,x)
        k2=dx * f(y+0.5*k1,x+0.5*dx)
        k3=dx * f(y+0.5*k2,x+0.5*dx)
        k4=dx * f(y+k3,x+dx)
        return y+(k1+2*k2+2*k3+k4)/6
    
    def fx(self,x,t):
        return fx.vel

    def fy(self,y,t):
        return fy.vel-9.8*t

    def step(self,dt):
        self.x=self.rk4(self.x,self.t,dt,fx)
        self.y=self.rk4(self.y,self.t,dt,fy)
        self.t+=dt
        return (self.x+np.random.randn()+self.noise[0],self.y+np.random.randn()+self.noise[1])

