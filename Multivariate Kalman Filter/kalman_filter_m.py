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
	'state_transition' is the state transition function (see Designing F)
	'measurement_transition' is the measurement function (see Designing H)
	'control_transition' is the control input function (see Designing B)
	'process_var' is the variance in the state variable
	'sensor_var' is the variance in the sensor readings
	'measurements' is the final value of the state variable after timestep dt
	'control_input' is the control input
	"""
	def __init__(self,state_variables,state_transition,measurement_transition,control_transition,process_var,sensor_var,measurements,control_input):
		
		self.state_variables=state_variables #state variable
		self.dim_x=len(self.state_variables.mean()) #Number of state variables 
		
		self.x = np.array([i for i in self.state_variables.mean()]).T #State
		self.P = np.diag(self.state_variables.variance())	#Prior Covariance
		self.F=state_transition #state transition function
		self.H=measurement_transition #measurement function 
		
		self.B=control_transition #control input function
		self.u=control_input #control input
		self.dim_u=len(control_input) #size of control input

		self.Q=process_var # process covariance (noise)
		self.R=sensor_var # measurement covariance.

		self.measurements=measurements
		self.dim_z=len(measurements) #No of measurement inputs

		self.S=np.zeros((self.dim_z, self.dim_z))
		self.K=np.zeros((self.dim_x, self.dim_z))
		self.y=np.zeros((self.dim_z, 1))

		self.ss=[]
		self.yy=[]
		self.xs,self.covs=self.filter()

	def predict(self):
		self.x = np.dot(self.F,self.x)+np.dot(self.B,self.u)
		self.P = np.dot(np.dot(self.F,self.P),self.F.T)+self.Q
		return self.x,self.P

	def update(self,z):
		self.S = np.dot(np.dot(self.H, self.P),(self.H.T)) + self.R 	# System Uncertainity
		self.K = np.dot(np.dot(self.P, self.H.T),np.linalg.inv(self.S))	# Kalman Gain
		self.y = z - np.dot(self.H, self.x)								# Residual
		self.ss.append(self.S)
		self.yy.append(self.y)
		I=np.eye(self.dim_x)
		self.x =  self.x + np.dot(self.K, self.y)		
		#self.P = self.P - np.dot(np.dot(K, self.H),self.P)
		
		self.P = np.dot(np.dot(I-np.dot(self.K,self.H),self.P),(I-np.dot(self.K,self.H)).T) + np.dot(np.dot(self.K,self.R),self.K.T) 
		#Accounts for floating point errors 
		# (I-KH)P(I-KH).T + KRK.T
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
		dx = np.diff(self.xs[:, 0], axis=0)
		dx=np.insert(dx,0,self.xs[0,1])
		if self.measurements.shape[1]==1:
			print(f'  Measurement\t Prediction\t\t Variance\t')

			for i in range(0,len(self.measurements)):
				print(f'\t{dx[i]:6.3f}\t',end='\t')
				print(f'\t{np.round(self.xs[i,1],3):6.3f}\t\t   {np.round(sqrt(self.covs[i][1,1]),3):6.3f}',end='\t')
		elif self.measurements.shape[1]==2:
			print(f'  Measurement1\t   Prediction1\t  Variance1\t  Measurement2\t  Prediction2\t  Variance2\t')
			for i in range(0,len(self.measurements)):
				print(f'\t{dx[i]:6.3f}\t',end='\t')
				print(f'\t{np.round(self.xs[i,1],3):6.3f}\t\t   {np.round(sqrt(self.covs[i][1,1]),3):6.3f}',end='\t')
				print(f'\t{dx[i]:6.3f}\t',end='\t')
				print(f'\t{np.round(self.xs[i,3],3):6.3f}\t\t   {np.round(sqrt(self.covs[i][3,3]),3):6.3f}')
		elif self.measurements.shape[1]==3:
			print(f'  Measurement1\t   Prediction1\t  Variance1\t  Measurement2\t  Prediction2\t  Variance2\t  Measurement3\t  Prediction3\t  Variance3\t')
			for i in range(0,len(self.measurements)):
				print(f'\t{dx[i]:6.3f}\t',end='\t')
				print(f'\t{np.round(self.xs[i,1],3):6.3f}\t\t   {np.round(sqrt(self.covs[i][1,1]),3):6.3f}',end='\t')
				print(f'\t{dx[i]:6.3f}\t',end='\t')
				print(f'\t{np.round(self.xs[i,3],3):6.3f}\t\t   {np.round(sqrt(self.covs[i][3,3]),3):6.3f}')
				print(f'\t{dx[i]:6.3f}\t',end='\t')
				print(f'\t{np.round(self.xs[i,5],3):6.3f}\t\t   {np.round(sqrt(self.covs[i][5,5]),3):6.3f}')


	def toPlot(self):
		plt.figure(1)
		self.measurements=self.measurements.reshape(-1,1)
		self.plotMeasurements(self.measurements)
		self.plotFilter(self.xs)
		plt.title('Kalman Filter')
		plt.legend(loc=4)
		plt.grid()
		

		# plt.figure(2)
		# self.plotFilter2()
		# plt.title('State Variable 2')
		# plt.legend(loc=4)
		# plt.grid()

		# y_pred1,y_pred2=self.getVariance(self.covs)
		
		# plt.figure(3)
		# self.plotVariance(y_pred1)
		# plt.title('Variance 1')
		# plt.grid()

		# plt.figure(4)
		# self.plotVariance(y_pred2)
		# plt.title('Variance 2')
		# plt.grid()
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

		x = np.arange(len(y))
		plt.plot(x,y[:,0], color='C0', label='Filter')

		if y.shape[1]==4:
			plt.plot(y[:,0],y[:,2], color='C0', label='Filter')
		elif y.shape[1]==6:
			plt.plot(y[:,0],y[:,2],y[:,4], color='C0', label='Filter')
		elif y.shape[1]>6:
			print("Shape out of bounds, edit plotFilter func()")
		
	def plotFilter2(self): #plots second state variable
		dx = np.diff(self.xs[:, 0], axis=0)
		y=self.xs[:,1]
		x=np.arange(len(y))
		plt.scatter(range(1, len(dx) + 1), dx, facecolor='none',edgecolor='k', lw=2, label='Measurement')
		plt.plot(x,y.T, color='C0', label='Filter')

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

		print(f'control input function: \n{self.B}')
		print(f'control input: {self.u}')
		print(f'size of control input: {self.dim_u}\n')

		print(f'process covariance (noise) : \n{self.Q}')
		print(f'sensor variance (sensor noise): \n{self.R}\n')

		print(f'number of measurements: {self.dim_z}\n')
		print(f'measurements:\n {self.measurements}')
		
	def out(self,doPrint=False):
		if doPrint:
			print(f'output readings are: \n{self.xs}\n')
			print(f'output covariances are: \n{self.covs}\n')
		return (self.xs,self.covs)