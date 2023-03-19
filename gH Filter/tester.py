from gh_filter import gH_filter
import numpy as np
import matplotlib.pyplot as plt

def genData(x0, dx, count, noise_factor):
	return [x0 + dx * i + np.random.randn() * noise_factor for i in range(count)]

def accData(x0, dx, count, noise_factor, accel = 0):
	zs = []
	for i in range(count):
		zs.append(x0 + dx * i + np.random.randn() * noise_factor)
		dx += accel
	return zs

def main():
	x0 = 160.0 #Init estimate
	dx = 1 #change rate
	
	g = 0.6 
	
	"""
		setting g~1, signal exactly follows the measurements implies NO NOISE REJECTION
		setting g~0, signal ignore measurements and follows prediction, implies noise filterd but also LEGITIMATE CHANGE IN SIGNAL
	
	"""

	h = 0.66
	
	"""
		setting h~1, signal reacts to change rate RAPIDLY, signal amplitude is less and settles on measurements fast
		setting h~0, signal reacts to change less SLOWLY, signal amplitude is large and settles on measurements slowly
	
	"""

	dt = 1	#time step

	#LESS NOISE
	measurements = genData(x0, dx, count=30, noise_factor=1)
	gh=gH_filter(measurements, x0, dx, g, h, dt, plot=True)
	#gh.toString()
	
	#MORE NOISE
	bad_measurements = genData(x0 = 5, dx = 2, count = 100, noise_factor = 10)
	gh_bad = gH_filter(bad_measurements, x0, dx, g, h, dt, plot=True)
	#gh_bad.toString()

	#EXTREME NOISE
	extreme_measurements = genData(x0 = 5, dx = 2, count = 100, noise_factor = 100)
	gh_extreme = gH_filter(extreme_measurements, x0, dx, g, h, dt, plot=True)
	#gh_extreme.toString()

	#EFFECT OF ACCELERATION
	accel_data = accData(x0=10, dx=0, count=20, noise_factor=0, accel=2)
	gh_accel = gH_filter(accel_data, x0, dx, g, h, dt, plot=True)

	#gh.Algorithm()
if __name__ == "__main__":
	main()
