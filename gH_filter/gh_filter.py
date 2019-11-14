import numpy as np
import matplotlib.pyplot as plt

class gH_filter:
	"""
	Performs g-h filter on 1 state variable with a fixed g and h value

	'data' contains the data to be filtered
	'x0' is the intial value of the state variable
	'dx' is the initial change rate of the state variable
	'g' is the g scale factor
	'h' is the h scale factor
	'dt' is the time step
	'plot' boolean variable that turns plotting results on/off
	"""

	def __init__(self,data,x0,dx,g,h,dt,plot=False):
		self.data=data
		self.x0=x0
		self.dx=dx
		self.g=g
		self.h=h
		self.dt=dt
		self.results=self.filter(self.data,self.x0,self.dx,self.g,self.h,self.dt)
		if plot:
			self.plotResults(self.data,self.results,self.x0)

	def filter(self,data,x0,dx,g,h,dt):
		results=[]
		x_est=self.x0
		for z in self.data:
			#prediction
			x_pred=x_est+(self.dx*self.dt)
			self.dx=self.dx

			#update
			residual = z - x_pred
			self.dx+=self.h*(residual)/self.dt
			x_est=x_pred+self.g*residual
			results.append(x_est)
		return np.array(results)

	def toString(self):
		print(f'final estimates are \n{np.array(self.results)}')


	def plotResults(self,data,ests,x0):
		if type(data)==list:
			size=len(self.data)
		elif type(data)==np.ndarray:
			size=data.shape[0]
		x=np.linspace(1,size,size)
		y=np.linspace(x0,ests[-1],size)
		plt.scatter(x,data,label='Measurements')
		plt.plot(x,ests,'r-',label='Estimate')
		#plt.plot(x,y,'k--',label='Actual')
		plt.grid()
		plt.title('GH Filter')
		plt.legend(loc=4)
		plt.show()