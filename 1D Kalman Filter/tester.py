import numpy as np
from kalman_filter import *
import random
from math import sqrt,pi
np.set_printoptions(precision=2, suppress=True, linewidth=60)
random.seed(13)


class createPoints(object):
    def __init__(self, x0=0, velocity=1,
                 sensor_var=0.0,
                 process_var=0.0):
        """ x0 : initial position
            velocity: (+=right, -=left)
            sensor_var: variance in measurement m^2
            process_var: variance in process (m/s)^2
        """
        self.x = x0
        self.velocity = velocity
        self.sensor_var = sensor_var
        self.process_var = process_var

    def move(self, dt=1.0):
        """Compute new position in dt seconds."""
        dx = self.velocity + np.random.randn()*sqrt(self.process_var)
        self.x += dx * dt

    def sense_position(self):
        """ Returns measurement of new position in meters."""
        measurement = self.x + np.random.randn()*sqrt(self.sensor_var)
        return measurement

    def move_and_sense(self):
        """Change position, and return measurement of new position in meters"""
        self.move()
        return self.sense_position()

def main():
	process_var=1 
	sensor_var=2 
	velocity=1
	dt=1 
	doPrint=True
	x = Gaussian(0., 20.**2)

	pts=createPoints(x.mean(),velocity,sensor_var,process_var)
	measurements=[]
	pos=[]
	for i in range(0,10):
		measurements.append(pts.move_and_sense())
		pos.append(pts.x)
	
	
	
	kf=Kalman_Filter(x,velocity,dt,process_var,sensor_var,measurements,pos,doPrint)	



if __name__=="__main__":
	main()