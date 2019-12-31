import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=True)
from math import sqrt, pi
import random
from scipy.linalg import block_diag
random.seed(5)
from kalman_filter_m import *

"""
We will begin by tracking a robot in a 2D space with a sensor that supplies a noisy measurement of position in a 2D space
"""

class RobotMove(object):
	def __init__(self, pos=[0, 0], vel=[0, 0], noise_std=1):
		self.pos = pos
		self.vel = vel
		self.noise_std = noise_std
        
	def sensor(self):
		self.pos[0] += self.vel[0]
		self.pos[1] += self.vel[1]

		return [self.pos[0] + np.random.randn() * self.noise_std,
		self.pos[1] + np.random.randn() * self.noise_std]

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

if __name__=="__main__":
	#Initial Values
	pos=[4,3]
	vel=[2,1]
	noise_std= 1
	
	sensor = RobotMove(pos,vel,noise_std)

	dt=1

	#Initial State Variables
	x=np.zeros(4)
	P=np.array([500,500,500,500])
	state_variable=Gaussian(x,P)

	#Getting Measurements
	N=20
	ps=np.array([sensor.sensor() for i in range(N)])
	#plt.scatter(ps[:,0],ps[:,1],edgecolor='k',facecolor='None',lw=2)
	#plt.show()

	#Desiging State Transition Matrix	
	F=np.eye(4)
	F[0][1]=F[2][3]=dt
	
	#Designing Process Noise matrix
	process_var=0.001
	Q = generateWhiteNoise(4,dt,process_var)
	
	#Designing Measurement Function
	H=np.zeros((2,4))
	H[0,0]=H[1,2]=1

	#Designing Control Input
	B=[]
	u=[]

	#Designing Measurement Noise Matrix
	R=np.diag([5,5])

	#CREATING KALMAN FILTER
	mkf=Kalman_Filter_multi(state_variable,F,H,B,Q,R,ps,u)
	mkf.filter_details()
	# mkf.toString()
	# mkf.toPlot()