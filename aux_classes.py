import numpy as np
from scipy.linalg import block_diag

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