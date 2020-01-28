import numpy as np
import matplotlib.pyplot as plt
import math
import sys
sys.path.insert(0,'../')
from rt_kalman import *
from aux_classes import *

"""
First, we track a ball that moves through a vaccum. Then we try to track it in air, by adding wind,drag etc

#PROJECTILE EQUATIONS
y=(gt^2)/2 + vy*t + y0
x=vx*t + x0

vx=v*cos(theta)
vy=v*sin(theta)


STATE EQUATIONS
x=(1*x+dt*vx) + (0*y+0*vy)
vx=(0*x+1*vx) + (0*y+0*vy)
y=(0*x+0*vx) + (1*y+dt*vy)
vy=(0*x+0*vx) + (0*y+1*vy)
"""


def ball_kf(x,y,omega,v0,dt,r=0.5,q=0,u0=-9.8):
	kf=KalmanFilter(dim_x=4,dim_z=2,dim_u=1)
	#Initial 
	omega=math.radians(omega)
	vx=math.cos(omega)*v0
	vy=math.sin(omega)*v0
	#Choosing state variables
	kf.x=np.array([[x,vx,y,vy]]).T
	#State Transition Function
	kf.F=np.eye(4)
	kf.F[0,1]=dt
	kf.F[2,3]=dt
	#Control Input Function
	kf.B = np.array([[0., 0., 0., dt]]).T
	kf.u=u0
	#Measurement function
	kf.H=np.zeros((2,4))
	kf.H[0,0]=1
	kf.H[1,2]=1
	#Measurement Noise
	kf.R*=r
	#Process Noise
	kf.Q*=q #Since vaccum
	return kf

def track_ball_vaccum(dt):
	x,y=0,1
	theta=35
	v0=80
	g=9.8
	t=0
	xs,ys,zs=[],[],[]
	ball=BallTrajectory2D(x0=x,y0=x,vel=v0,theta=theta,noise=[0.2,0.2])	
	kf=ball_kf(x,y,theta,v0,dt)
	t=0
	xs,zs=[],[]
	while kf.x[2]>0: #y (height) above ground
		t+=dt
		x,y=ball.step(dt)
		z=np.array([[x,y]]).T
		zs.append(z)
		
		kf.update(z)
		xs.append([kf.x[0],kf.x[2]])
		kf.predict()
	
	kf.plotAll(zs,xs)

if __name__=="__main__":
	track_ball_vaccum(0.1)
