import numpy as np
from kalman_filter import *
import random
from math import sqrt,pi
np.set_printoptions(precision=2, suppress=True)
random.seed(13)

def main():
	"""
	a Kalman filter for a thermometer.
	The thermometer outputs a voltage that corresponds to the temperature that is being measured
	"""
	def volt(voltage, std):
		return voltage + (np.random.randn() * std)

	voltage_std = .13
	process_var = .05**2
	actual_voltage = 16.3
	temp_change = 0
	dt=1
	x = Gaussian(25, 1000)
	
	N=50
	measurements=[]

	for i in range(N):
		measurements.append(volt(actual_voltage,voltage_std))
	doPrint=False
	print(np.std(measurements))
	kf=Kalman_Filter(x,temp_change,dt,process_var,voltage_std**2,measurements,doPrint)
	kf.toPlot()


if __name__=="__main__":
	main()