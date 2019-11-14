import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
class Discrete_Bayes_Filter():
	"""
	prior - 
	kernel - 
	measurements - 
	z_prob - 
	hallway - 
	
	"""
	def __init__(self,prior,kernel,measurements,offset,z_prob,scale):
		self.prior=prior
		self.kernel=kernel
		self.measurements=measurements
		self.offset=offset
		self.z_prob=z_prob
		self.posterior=np.array([.1]*10)
		self.scale = scale
		self.priors,self.posteriors=self.filter()
	
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
			hall = self.z_prob / (1. - self.z_prob)
		except ZeroDivisionError:
			hall = 1e8
		likelihood = np.ones(len(self.scale))
		likelihood[self.scale==z] *= hall
		return likelihood

	def filter(self):
		priors=[]
		posteriors=[]	
		for i,z in enumerate(self.measurements):
			prior=self.predict(self.posterior)
			priors.append(prior)

			likelihood=self.lhScale(z)
			posterior=self.update(likelihood,prior)
			posteriors.append(posterior)
		return priors, posteriors

	def toString(self):
		print(f'the priors are \n{np.array(self.priors)}')
		print(f'the posteriors are \n{np.array(self.posteriors)}')

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
		self.barPlot(self.scale,c='k')
		self.barPlot(self.posteriors[i],ylim=(0,1.0))
		plt.axvline(i % len(self.scale), lw=5) 
		plt.show()

	def plotPriors(self,i):
		plt.title('Prior')
		self.barPlot(self.scale,c='k')
		self.barPlot(self.priors[i],ylim=(0,1.0))
		plt.axvline(i % len(self.scale), lw=5) 
		plt.show()
		
