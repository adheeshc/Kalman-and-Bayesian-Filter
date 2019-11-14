import numpy as np
import matplotlib.pyplot as plt

from gh_filter import gH_filter


"""
Problem Statement -  Design a filter for a train. Its position is expressed as its position on the track in relation to some fixed point which we say is 0 km. I.e., a position of 1 means that the train is 1 km away from the fixed point. Velocity is expressed as meters per second. We perform measurement of position once per second, and the error is +- 500 meters.

"""

def computeNewPosition(pos, vel, dt=1):
	return pos + (vel * dt)

def measurePosition(pos):
	return pos + np.random.randn()*500

def genData(pos,vel,count):
	zs=[]
	for i in range(0,count):
		pos=computeNewPosition(pos,vel)
		vel += 0.2
		zs.append(measurePosition(pos))
	return np.array(zs)

def plotData(zs):
	plt.plot(zs/1000)
	plt.xlabel('Time')
	plt.ylabel('km')
	plt.title('Train Positon')
	plt.grid()
	plt.show()

def main():
	pos=23*1000
	vel=15
	zs=genData(pos,vel,100)
	#plotData(zs)
	dx=15
	
	"""
	Choosing g and h
		Since its a train, it cannot change position very quickly, so H WILL BE SMALL. If the train never changes velocity we would make h extremely small to avoid having the filtered estimate unduly affected by the noise in the measurement. 
		We dont give much weight to measurements, so G WILL BE SMALL


	"""
	g=0.01
	h=0.001
	dt=1

	filt=gH_filter(zs,pos,dx,g,h,dt,plot=True)


if __name__=="__main__":
	main()

