import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
import random
np.set_printoptions(precision=2, suppress=True, linewidth=60)
random.seed(5)

class Discrete_Bayes_Filter():
	
	"""
	Performs bayes filter on 1 state variable given our original assumption and the sensor readings

	'prior' is initial belief of the state variable
	'kernel' is the kernel used
	'measurements' is the position moved at time t
	'offset' is the distance moved by the state variable
	'sensor_accuracy' is the accuracy of the sensor
	'scale' is the original track on which the state variable is evaluated
	
	'doPrint' is used to print results
	"""

	def __init__(self,prior,kernel,measurements,offset,sensor_accuracy,scale,doPrint=False):
		self.prior=prior
		self.kernel=kernel
		self.measurements=measurements
		self.offset=offset
		self.z_prob=sensor_accuracy
		self.posterior=self.normalize(prior[:])
		self.scale = scale
		self.priors,self.posteriors=self.filter(doPrint)
	
	def predict(self,pdf):
		return convolve(np.roll(pdf, self.offset), self.kernel, mode='wrap')
	
	def update(self,likelihood, prior):
		posterior = prior * likelihood
		return self.normalize(posterior)

	def normalize(self,pdf):
		pdf /= sum(np.asarray(pdf, dtype=float))
		return pdf

	def lhScale(self,z):
		try:
			temp = self.z_prob / (1. - self.z_prob)
		except ZeroDivisionError:
			temp = 1e8
		likelihood = np.ones(len(self.scale))
		likelihood[self.scale==z] *= temp
		return likelihood

	def sense(self,z):
		pos = z
		# insert random sensor error
		if random.random() > self.z_prob:
			if random.random() > 0.5:
				pos += 1
			else:
				pos -= 1
		return pos

	def filter(self,doPrint):
		priors=[]
		posteriors=[]	
		for i,z in enumerate(self.measurements):
			prior=self.predict(self.posterior)
			priors.append(prior)

			pos=self.sense(z)
			likelihood=self.lhScale(pos)
			
			self.posterior=self.update(likelihood,prior)
			posteriors.append(self.posterior)
			index = np.argmax(self.posterior)
			
			if doPrint:
				self.toString(i,z,pos,index)
		return priors,posteriors

	def toString(self,i,z,pos,index):
		print(f'time {i}: actual position {z}, sensed at {pos}')
		print(f'estimated position is {index} with confidence {(self.posterior[index]*100):.4f}%:')
		pass		

	def barPlot(self,pos,x=None,ylim=(0,1),c='#30a2da',title=None):
		ax = plt.gca()
		if x is None:
			x = np.arange(len(pos))
		ax.bar(x,pos,color=c)
		if ylim:
			plt.ylim(ylim)
		plt.xticks(np.asarray(x),x)
		if title is not None:
			plt.title(title)

	def plotPosterior(self,i):		
		plt.title('Posterior')
		self.barPlot(self.posteriors[i],ylim=(0,1.0))
		plt.axvline(self.measurements[i], lw=5) 
		plt.show()