from gh_filter import gH_filter
import numpy as np
import matplotlib.pyplot as plt

def genData(x0, dx, count, noise_factor):
	return [x0 + dx*i + np.random.randn()*noise_factor for i in range(count)]

def main():
	x0=160.0
	dx=1
	g=6/10
	h=2/3
	dt=1	

	#LESS NOISE
	measurements=genData(x0,dx,count=30,noise_factor=1)
	gh=gH_filter(measurements,x0,dx,g,h,dt,plot=True)
	#gh.toString()
	
	#MORE NOISE
	bad_measurements = genData(x0=5., dx=2., count=100, noise_factor=10)
	gh_bad=gH_filter(bad_measurements,x0,dx,g,h,dt,plot=True)
	#gh_bad.toString()

	#EXTREME NOISE
	extreme_measurements = genData(x0=5., dx=2., count=100, noise_factor=100)
	gh_extreme=gH_filter(extreme_measurements,x0,dx,g,h,dt,plot=True)
	#gh_extreme.toString()

if __name__=="__main__":
	main()
