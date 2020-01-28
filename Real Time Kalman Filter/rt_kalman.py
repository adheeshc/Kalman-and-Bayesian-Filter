import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)
from math import sqrt, pi
import random
import copy
np.random.seed(5)

class KalmanFilter():
	"""
	'state_variables' are the means and variances of the state variables
	'dt' is the timestep
	'state_transition' is the state transition function (see Designing F)
	'measurement_transition' is the measurement function (see Designing H)
	'control_transition' is the control input function (see Designing B)
	'process_var' is the variance in the state variable
	'sensor_var' is the variance in the sensor readings
	'measurements' is the final value of the state variable after timestep dt
	'control_input' is the control input
	"""
	def __init__(self,dim_x,dim_z,dim_u=0,u=None):
		self.dim_x = dim_x  #Number of state variables 
		self.dim_z = dim_z	#Number of measurement inputs
		self.dim_u = dim_u	#size of control input

		self.x = np.zeros((dim_x,1))	#state means
		self.P = np.eye(dim_x)			#Prior Covariance
		self.F = np.eye(dim_x)			#state transition function
		self.H = np.zeros((dim_z,dim_x))	#measurement function 
		
		self.B = None	#control input function
		self.u = u 		#control input
		
		self.Q = np.eye(dim_x) #process covariance (noise)
		self.R = np.eye(dim_z) #measurement covariance.

		self.z=np.array([[None]*self.dim_z]).T #Measurements
		
		self.S=np.zeros((dim_z, dim_z))	#System Uncertainity
		self.K=np.zeros((dim_x, dim_z))	#Kalman Gain
		self.y=np.zeros((dim_z, 1))		#Residual

		self.x_prior = self.x.copy()	#value after Predict()
		self.P_prior = self.P.copy()	#value after Predict()

		self.x_post = self.x.copy()		#value after Update()
		self.P_post = self.P.copy()		#value after Update()

	def predict(self):
		"""
		Predicts next state using KF state propagation equations
		"""
		self.Q=np.eye(self.dim_x)*self.Q
		if self.B is not None and self.u is not None:
			self.x = np.dot(self.F,self.x)+np.dot(self.B,self.u)
		else:
			self.x = np.dot(self.F,self.x)

		self.P = np.dot(np.dot(self.F,self.P),self.F.T)+self.Q
		self.x_prior = self.x.copy()
		self.P_prior = self.P.copy()

		return self.x,self.P

	def update(self,z):
		"""
		Add a new measurement to the KF
		"""
		if z is None:
			self.z=np.array([[None]*self.dim_z]).T
			self.x_post=self.x.copy()
			self.P_post=self.P.copy()
			self.y=np.zeros((dim_z, 1))
			return

		self.R=np.eye(self.dim_z)*self.R

		self.y = z - np.dot(self.H, self.x)								# Residual
		self.S = np.dot(np.dot(self.H, self.P),(self.H.T)) + self.R 	# System Uncertainity
		self.K = np.dot(np.dot(self.P, self.H.T),np.linalg.inv(self.S))	# Kalman Gain
		
		I=np.eye(self.dim_x)
		self.x =  self.x + np.dot(self.K, self.y)		
		#self.P = self.P - np.dot(np.dot(K, self.H),self.P)
		self.P = np.dot(np.dot(I-np.dot(self.K,self.H),self.P),(I-np.dot(self.K,self.H)).T) \
		+ np.dot(np.dot(self.K,self.R),self.K.T) 						#Accounts for floating point errors 
																		# (I-KH)P(I-KH).T + KRK.T
		self.z = copy.deepcopy(z)
		self.x_post = self.x.copy()
		self.P_post = self.P.copy()

	def plotAll(self,z,x,title='Kalman Filter'):
		plt.figure(1)
		
		z=np.array(z)
		z=z.reshape(z.shape[0],z.shape[1])

		x=np.array(x)
		x=x.reshape(x.shape[0],x.shape[1])
		self.plotMeasurements(z)
		self.plotFilter(x)
		plt.title(title)
		plt.legend(loc=4)
		plt.grid()
		plt.show()

	
	def plotMeasurements(self,y):
		if y.shape[1]==1:
			x=np.arange(len(y))
			plt.scatter(x, y, edgecolor='k', facecolor='none',lw=2, label='Measurements')
		elif y.shape[1]==2:
			plt.scatter(y[:,0], y[:,1], edgecolor='k', facecolor='none',lw=2, label='Measurements')
		elif y.shape[1]==3:
			plt.scatter(y[:,0], y[:,1], y[:,2], edgecolor='k', facecolor='none',lw=2, label='Measurements')
		else:
			print("Shape out of bounds, check plotMeasurements func()")

	def plotFilter(self,y): #Plots first state variable
		if y.shape[1]==1:
			x=np.arange(len(y))
			plt.plot(x, y, edgecolor='k', facecolor='none',lw=2, label='Measurements')
		if y.shape[1]==2:
			plt.plot(y[:,0],y[:,1], color='C0', label='Filter')
		elif y.shape[1]==3:
			plt.plot(y[:,0],y[:,1],y[:,2], color='C0', label='Filter')
		else:
			print("Shape out of bounds, edit plotFilter func()")
	
	def plotResiduals(self,x,stds):
		if self.dim_x==1:
			res1 = x[:,0] - self.xs[:,0]
			plt.plot(res1)
			plt.title(f'Residuals for State Variable 1 ({stds}\u03C3)')
			plt.xlabel('time(sec)')
			self.plot_residual_limits(self.covs.flatten(),stds)
		elif self.dim_x==2:
			y_pred1,y_pred2=self.getVariance(self.covs)
			plt.figure(1)
			res1 = x[:,0] - self.xs[:,0]
			plt.plot(res1)
			plt.title(f'Residuals for State Variable 1 ({stds}\u03C3)')
			plt.xlabel('time(sec)')
			self.plot_residual_limits(y_pred1,stds)
			plt.figure(2)
			res2 = x[:,1] - self.xs[:,1]
			plt.plot(res2)
			plt.title(f'Residuals for State Variable 2 ({stds}\u03C3)')
			plt.xlabel('time(sec)')
			self.plot_residual_limits(y_pred2,stds)
		elif self.dim_x==3:
			y_pred1,y_pred2=self.getVariance(self.covs)
			plt.figure(1)
			res1 = x[:,0] - self.xs[:,0]
			plt.plot(res1)
			plt.title(f'Residuals for State Variable 1 ({stds}\u03C3)')
			plt.xlabel('time(sec)')
			self.plot_residual_limits(y_pred1,stds)
			plt.figure(2)
			res2 = x[:,1] - self.xs[:,1]
			plt.plot(res2)
			plt.title(f'Residuals for State Variable 2 ({stds}\u03C3)')
			plt.xlabel('time(sec)')
			self.plot_residual_limits(y_pred2,stds)
		else:
			print("Shape out of bounds, edit plot_residuals func()")
		plt.show()

	def plot_residual_limits(self,Ps, stds): #plots standand deviation given in Ps 
	    std = np.sqrt(np.abs(Ps)) * stds
	    plt.plot(-std, color='k', ls=':', lw=2)
	    plt.plot(std, color='k', ls=':', lw=2)
	    plt.fill_between(range(len(std)), -std, std,facecolor='#ffff00', alpha=0.3)

	def getVariance(self,y):
		y_pred1=[]
		y_pred2=[]
		for i in range(len(y)):
			y_pred1.append(y[i][0,0])
			y_pred2.append(y[i][1,1])
		return y_pred1,y_pred2

	def plotVariance(self,y):
		x=np.arange(len(y))
		plt.plot(x,y, color='C0')

	def Algorithm(self):
		print('PREDICT STEP')
		print('Initial Mean : x_=F*x+B*u')
		print('Initial Covariance : P_=F*P*F.T + Q')
		print()
		print('UPDATE STEP')
		print('System Uncertainty : S=H*P_*H.T + R')
		print('Kalman Gain : K=P*H.T*inv(S)')
		print('Residual : y=z-H*x_')
		print('Covariance Update : P=(1-K*H)*P_')

	def filter_details(self):

		print(f'number of state variables: {self.dim_x}')
		print(f'Means: {self.x}')
		print(f'Covariance: \n{self.P}\n')

		print(f'state transition function: \n{self.F}')
		print(f'measurement transition function: \n{self.H}\n')

		print(f'size of control input: {self.dim_u}\n')
		print(f'control input function: \n{self.B}')
		print(f'control input: {self.u}')

		print(f'process covariance (noise) : \n{self.Q}')
		print(f'sensor variance (sensor noise): \n{self.R}\n')

		print(f'number of measurements: {self.dim_z}\n')
		print(f'measurements:\n {self.z}')