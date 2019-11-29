import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)
from math import sqrt, pi
import random
random.seed(5)

class Gaussian:
	def __init__(self,mu,var):
		self.mu=mu
		self.var=var

	def mean(self):
		return self.mu

	def variance(self):
		return self.var

class Kalman_Filter_multi():
	"""
	'state_variables' are the means and variances of the state variables
	'dt' is the timestep
	'process_var' is the variance in the state variable
	'sensor_var' is the variance in the sensor readings
	'measurements' is the final value of the state variable after timestep dt
	'pos' is the true value of the state variable (without sensor var)

	"""
	def __init__(self,state_variables,dt,process_var,sensor_var,measurements,pos):
		
		self.state_variables=state_variables #state variable
		self.dim_x=len(self.state_variables.mean()) #size of the state vector 
		
		self.x = np.array([i for i in self.state_variables.mean()]).T
		self.P = np.diag(self.state_variables.variance())
		

		self.dt=dt #time step in sec
		self.F=np.array([[1, dt],[0, 1]]) #state transition function (NEEDS T0 BE CHANGED AS REQD)
		self.H=np.array([[1., 0.]]) #measurement function (NEEDS T0 BE CHANGED AS REQD)
		
		self.Q=process_var # process covariance (noise)
		self.R=sensor_var # measurement covariance.

		self.measurements=measurements
		self.pos=pos

		self.xs,self.covs=self.filter()


	def predict(self):
		self.x = np.dot(self.F,self.x)
		self.P = np.dot(np.dot(self.F,self.P),self.F.T)+self.Q
		return self.x,self.P

	def update(self,z):
		S = np.dot(np.dot(self.H, self.P),(self.H.T)) + self.R 	# System Uncertainity
		K = np.dot(np.dot(self.P, self.H.T),np.linalg.inv(S))	# Kalman Gain
		y = z - np.dot(self.H, self.x)							# Residual
		self.x =  self.x + np.dot(K, y)							# posterior
		self.P = self.P - np.dot(np.dot(K, self.H),self.P)		# posterior variance
		return self.x,self.P

	def filter(self):
		covs=[]
		xs=[]
		for i, z in enumerate(self.measurements):
			
			#Predict
			self.predict()
			
			#Update
			self.update(z)
			
			xs.append(self.x)
			covs.append(self.P)

		return np.array(xs),np.array(covs)

	def toString(self):
		print(f'  Measurement\t Prediction\t\t Variance\t')
		for i in range(0,len(self.measurements)):
			print(f'\t{self.xs[i,0]:6.3f}\t',end='\t')
			print(f'\t{np.round(self.pos[i],3)}\t\t   {np.round(sqrt(self.covs[i][0,0]),3)}')

	def toPlot(self):
		plt.figure(1)
		self.plotMeasurements(self.measurements)
		self.plotFilter()
		plt.title('Multivariate Kalman Filter')
		plt.legend(loc=4)
		plt.grid()
		plt.show()
	
	def plotMeasurements(self,y):
		x = np.arange(len(y))
		plt.scatter(x, y, edgecolor='k', facecolor='none',lw=2, label='Measurements')

	def plotFilter(self):
		y=self.xs[:,0]
		x = np.arange(len(y))
		
		std = np.sqrt(self.covs[:,0,0])
		std_top = np.minimum(self.pos+std, [len(self.measurements) + 10])
		std_btm = np.maximum(self.pos-std, [-50])

		std_top = self.pos + std
		std_btm = self.pos - std

		plt.plot(x,y.T, color='C0', label='Filter')
		plt.plot(x,self.pos,linestyle=':', color='k', lw=2)
		plt.plot(x, std_top, linestyle=':', color='k', lw=1,alpha=0.4)
		plt.plot(x, std_btm, linestyle=':', color='k', lw=1,alpha=0.4)
		plt.fill_between(x, std_btm, std_top,facecolor='yellow', alpha=0.2)
