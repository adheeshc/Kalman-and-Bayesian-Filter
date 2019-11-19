from discrete_bayes_filter import Discrete_Bayes_Filter
import numpy as np
import matplotlib.pyplot as plt

def main():
	prior = np.array([.9] + [0.01] * 9)
	kernel=(.1,.8,.1)
	z_prob=0.9
	offset=4
	scale=np.array([0,1,2,3,4,5,6,7,8,9])
	measurements = [4,8,3,7]
	doPrint=True


	dbf = Discrete_Bayes_Filter(prior, kernel, measurements,offset, z_prob,scale,doPrint)
	for i in range(len(measurements)):
		dbf.plotPosterior(i)
	

if __name__=="__main__":
	main()
