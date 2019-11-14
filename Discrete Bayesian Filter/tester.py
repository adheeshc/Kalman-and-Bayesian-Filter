from discrete_bayes_filter import Discrete_Bayes_Filter
import numpy as np
import matplotlib.pyplot as plt

def main():
	prior = np.array([.1] * 10)
	kernel=(.1,.8,.1)
	z_prob=1.0
	scale=np.array([1,1,0,0,0,0,0,0,1,0])
	offset=1
	
	# measurements with no noise
	#measurements = [scale[i % len(scale)] for i in range(50)]
	

	measurements = [1, 0, 1, 0, 0, 1, 1]

	dbf = Discrete_Bayes_Filter(prior, kernel, measurements,offset, z_prob,scale)
	dbf.plotPosterior(6)
	dbf.plotPriors(6)
	

if __name__=="__main__":
	main()
