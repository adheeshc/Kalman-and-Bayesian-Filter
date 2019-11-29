import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
#np.set_printoptions(precision=2, suppress=True)
from math import sqrt, pi
import random
random.seed(5)
from kalman_filter_m import *

class createPoints(object):
    def __init__(self, x0=0, velocity=1,sensor_var=0.0,process_var=0.0):
        """ x0 : initial position
            velocity: (+=right, -=left)
            sensor_var: variance in measurement m^2
            process_var: variance in process (m/s)^2
        """
        self.x = x0
        self.velocity = velocity
        self.sensor_var = sensor_var
        self.process_var = process_var

    def move(self, dt=1): #Compute new position in dt seconds.
        dx = self.velocity + np.random.randn()*sqrt(self.process_var)
        self.x += dx * dt
        return self.x

    def sense_position(self): #Returns measurement of new position in meters.
        measurement = self.x + np.random.randn()*sqrt(self.sensor_var)
        return measurement

    def move_and_sense(self): #Change position, and return measurement of new position in meters
        self.move()
        return self.sense_position()

	
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

	process_var=scipy.linalg.block_diag(*[Q]*1) * process_var

	return process_var

if __name__=="__main__":
	state_var=Gaussian([10,4.5],[500,49])
	num_state_var=len(state_var.mean())
	dt = 1

	sensor_var = 10
	process_var = 0.01
	pts=createPoints(sensor_var=sensor_var,process_var=process_var)

	process_var=generateWhiteNoise(num_state_var,dt,process_var)
	
	N=20
	measurements=[]
	pos=[]
	for i in range(0,N):
		measurements.append(pts.move_and_sense())
		pos.append(pts.move())
	measurements=np.array(measurements)
	pos=np.array(pos)
    
	kfm=Kalman_Filter_multi(state_var, dt, process_var, sensor_var, measurements,pos)
	#kfm.toString()
	kfm.toPlot()
