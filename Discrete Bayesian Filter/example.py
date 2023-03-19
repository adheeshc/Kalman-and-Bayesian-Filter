import numpy as np
import matplotlib.pyplot as plt
from discrete_bayes_filter import Discrete_Bayes_Filter
import random
from scipy.ndimage import convolve
np.set_printoptions(precision=2, suppress=True, linewidth=60)
random.seed(5)


class Train(object):
	def __init__(self, track_len, kernel=[1.], sensor_accuracy=.9):
		self.track_len = track_len
		self.pos = 0
		self.kernel = kernel
		self.sensor_accuracy = sensor_accuracy

	def move(self, distance=1):
		"""
		move in the specified direction with some small chance of error
		"""
		self.pos += distance
		# insert random movement error according to kernel
		r = random.random()
		s = 0
		offset = -(len(self.kernel) - 1) / 2
		for k in self.kernel:
			s += k
			if r <= s:
				break
			offset += 1
		self.pos = int((self.pos + offset) % self.track_len)
		return self.pos

	def sense(self):
		pos = self.pos
		# insert random sensor error
		if random.random() > self.sensor_accuracy:
			if random.random() > 0.5:
				pos += 1
			else:
				pos -= 1
		return pos

def train_filter(iterations, kernel, sensor_accuracy, offset, do_print=True):
	scale = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	prior = np.array([.9] + [0.01]*9)
	posterior = prior[:]
	measurements=[]

	robot = Train(len(scale), kernel, sensor_accuracy)
	for i in range(iterations):
		measurements.append(robot.move(distance=offset))
	
	dbf=Discrete_Bayes_Filter(prior,kernel,measurements,offset,sensor_accuracy,scale,doPrint=True)
	for i in range(iterations):
		dbf.plotPosterior(i)
	#dbf.Algorithm()

if __name__=="__main__":
	train_filter(4,kernel=[.1, .8, .1], sensor_accuracy=.9,offset=4, do_print=True)
