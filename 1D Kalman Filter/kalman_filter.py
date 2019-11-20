import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True, linewidth=60)
from math import sqrt, pi
import random
random.seed(5)


class Gaussian():
	"""
	Creates a 1D gaussian given mean and variance

	'mu' is the mean
	'sigma' is the variance

	"""
	def __init__(self,mu,sigma):
		self.mu=mu
		self.sigma=sigma
	
	def mean(self):
		return self.mu

	def variance(self):
		return self.sigma

	def pdf(self):
		return (1/sqrt(2*pi*self.sigma))*exp((-(x-self.mu)**2)/(2*self.sigma))

class Kalman_Filter():
	def __init__(self,x,velocity,dt,process_var,sensor_var,measurements,pos,doPrint=False):
		self.posterior = x #
		self.velocity=velocity #velocity of state variable
		self.dt=dt #time step in sec
		self.process_var=process_var #variance in state variable
		self.sensor_var=sensor_var  #variance in sensor
		self.measurements=measurements #position of state variable
		self.pos=pos
		self.process_model=Gaussian(self.velocity*self.dt,self.process_var)  #displacement of state variable with time
		self.priors,self.posteriors=self.filter(doPrint)
		
	def predict(self,pos,movement):
		return Gaussian(pos.mean()+movement.mean(),pos.variance() + movement.variance())

	def update(self,prior,likelihood):
		posterior=self.gaussianMultiply(likelihood,prior)
		return posterior

	def gaussianMultiply(self,g1,g2):
		mean = (g1.variance()*g2.mean()+g2.variance()*g1.mean()) / (g1.variance()+g2.variance())
		variance = (g1.variance()*g2.variance()) / (g1.variance()+g2.variance())
		return Gaussian(mean,variance)

	def filter(self,doPrint):
		priors=[]
		posteriors=[]
		for i, z in enumerate(self.measurements):
			prior=self.predict(self.posterior,self.process_model)
			#priors.append(prior)
			
			likelihood=Gaussian(z,self.sensor_var)
			
			self.posterior=self.update(prior,likelihood)
			#posteriors.append(self.posterior)
			index = np.argmax(self.posterior)
			
			if doPrint:
				self.print_gh(prior,self.posterior,z)
		
		return priors,posteriors

	def print_gh(self,predict,update, z):
		predict_template = '{: 7.3f} {: 8.3f}'
		update_template = '{:.3f}\t{: 7.3f} {: 7.3f}'
		
		print(predict_template.format(predict[0], predict[1]),end='\t')
		print(update_template.format(z, update[0], update[1]))

	def toString(self,i):
		print(f'final estimate: {self.posterior.mean():10.3f}')
		#print(f'actual final position: {self.pos()[i]:10.3f}')

		


