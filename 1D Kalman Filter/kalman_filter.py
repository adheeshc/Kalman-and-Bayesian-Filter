import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)
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
	"""

	'x' is the initialize estimate
	'change' is the change of the state variable
	'dt' is the time step in secs
	'process_var' is the variance in the state variable
	'sensor_var' is the variance in the sensor readings
	'pos' is the upper and lower bound of the state
	'measurements' is the final value of the state variable after timestep dt
	'doPrint' is toggle for printing values

	"""
	def __init__(self,x,change,dt,process_var,sensor_var,measurements,doPrint=False):
		self.posterior = x #posterior
		self.change=change #change of state variable
		self.dt=dt #time step in sec
		self.process_var=process_var #variance in state variable
		self.sensor_var=sensor_var  #variance in sensor
		self.measurements=measurements #value of state variable
		
		x0=min(measurements)-np.std(measurements)*2
		x1=max(measurements)+np.std(measurements)*2
		self.pos=[x0,x1]

		self.process_model=Gaussian(self.change*self.dt,self.process_var)  #displacement of state variable with time
		self.priors,self.posteriors=self.filter(doPrint)
		
	def predict(self,pos,movement):
		
		x=pos.mean() 
		P=pos.variance()
		dx=movement.mean()
		Q=movement.variance() #process noise

		x = x + dx
		P = P + Q

		return Gaussian(x,P)

	def update(self,prior,likelihood):
		x=prior.mean()
		P=prior.variance() 		#variance of the state
		z=likelihood.mean()		#measurement 
		R=likelihood.variance() #measurement noise

		y = z - x        # residual
		K = P / (P + R)  # Kalman gain

		x = x + K*y      # posterior
		P = (1 - K) * P  # posterior variance
		return Gaussian(x, P)

	def gaussianMultiply(self,g1,g2):
		mean = (g1.variance()*g2.mean()+g2.variance()*g1.mean()) / (g1.variance()+g2.variance())
		variance = (g1.variance()*g2.variance()) / (g1.variance()+g2.variance())
		return Gaussian(mean,variance)

	def filter(self,doPrint):
		priors=[]
		posteriors=[]
		if doPrint:
			print(f'\t\t   PREDICT\t\t\t\t\t\t\t\t UPDATE')
			print(f'\t prior\t\t variance\t\t measurement\tposterior\tvariance')
		for i, z in enumerate(self.measurements):
			prior=self.predict(self.posterior,self.process_model)
			priors.append(prior)
			
			likelihood=Gaussian(z,self.sensor_var)
			
			self.posterior=self.update(prior,likelihood)
			posteriors.append(self.posterior)
			
			if doPrint:
				self.toString(prior,self.posterior,z)
		if doPrint:
			print()
		print(f'Final Position: {self.measurements[-1]:7.3f}')
		print(f'Final Estimate: {self.posterior.mean():7.3f}')
		
		return priors,posteriors

	def toString(self,predict,update,z):
		print(f'\t{predict.mean():6.3f}\t\t{predict.variance():8.3f}',end='\t')
		print(f'\t\t{z:.3f} \t\t{update.mean(): 7.3f} \t{update.variance(): 7.3f}')
	
	def toPlot(self):
		plt.figure(1)
		self.plotMeasurements(self.measurements)
		self.plotFilter(self.posteriors)
		self.plotPredictions(self.priors)
		plt.title('Kalman Filter')
		plt.ylim(self.pos[0], self.pos[1])
		plt.legend(loc=4)
		
		plt.figure(2)
		plt.plot([i.variance() for i in self.posteriors])
		plt.title('Variance')
		plt.show()
	
	def plotMeasurements(self,y):
		x = np.arange(len(y))
		plt.scatter(x, y, edgecolor='k', facecolor='none',lw=2, label='Measurements')

	def plotFilter(self,y):
		x = np.arange(len(y))

		y=[]
		y.append([i.mean() for i in self.posteriors])
		y=np.array(y)
		y=y[0]
		
		var=[]
		var.append([i.variance() for i in self.priors])
		var=np.array(var)
		var=var[0]
		
		std = np.sqrt(var)
		std_top = y+std
		std_btm = y-std

		plt.plot(x,y.T, color='C0', label='Filter')
		plt.plot(x, std_top, linestyle=':', color='k', lw=2)
		plt.plot(x, std_btm, linestyle=':', color='k', lw=2)

		plt.fill_between(x, std_btm, std_top,facecolor='yellow', alpha=0.2)

	def plotPredictions(self,y):
		x=np.arange(len(y))
		
		y=[]
		y.append([i.mean() for i in self.priors])
		y=np.array(y)
		y=y[0]

		plt.scatter(x, y, marker='v', s=40, edgecolor='r',facecolor='None', lw=2, label="Priors")

	def Algorithm(self):
		print(f"""
		Initialization

		1. Initialize the state of the filter
		2. Initialize our belief in the state

		Predict

		1. Use system behavior to predict state at the next time step
		2. Adjust belief to account for the uncertainty in prediction

		Update

		1. Get a measurement and associated belief about its accuracy
		2. Compute residual between estimated state and measurement
		3. Compute scaling factor based on whether the measurement or prediction is more accurate
		4. set state between the prediction and measurement based on scaling factor
		5. update belief in the state based on how certain we are in the measurement
		""")