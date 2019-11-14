import numpy as np
import matplotlib.pyplot as plt
from discrete_bayes_filter import Discrete_Bayes_Filter
import random
from scipy.ndimage.filters import convolve
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


def predict(pdf,offset,kernel):
	return convolve(np.roll(pdf, offset), kernel, mode='wrap')
	
def update(likelihood, prior):
	posterior = prior * likelihood
	return normalize(posterior)

def normalize(pdf):
	pdf /= sum(np.asarray(pdf, dtype=float))
	return pdf

def lhScale(hall, z, z_prob):
	try:
		scale = z_prob / (1. - z_prob)
	except ZeroDivisionError:
		scale = 1e8
	likelihood = np.ones(len(hall))
	likelihood[hall==z] *= scale
	return likelihood

def barPlot(pos,x=None,ylim=(0,1),c='#30a2da',title=None):
	ax = plt.gca()
	if x is None:
		x = np.arange(len(pos))
	ax.bar(x,pos,color=c)
	if ylim:
		plt.ylim(ylim)
	plt.xticks(np.asarray(x),x)
	if title is not None:
		plt.title(title)	
	plt.show()


def train_filter(iterations, kernel, sensor_accuracy, move_distance, do_print=True):
	track = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
	prior = np.array([.9] + [0.01]*9)
	posterior = prior[:]
	normalize(prior)

	robot = Train(len(track), kernel, sensor_accuracy)
	for i in range(iterations):
		# move the robot and
		robot.move(distance=move_distance)



		# peform prediction
		prior = predict(posterior, move_distance, kernel)       

		#  and update the filter
		m = robot.sense()
		likelihood = lhScale(track, m, sensor_accuracy)
		posterior = update(likelihood, prior)
		index = np.argmax(posterior)

		if do_print:
			print(f'time {i}: pos {robot.pos}, sensed {m}, at position {track[robot.pos]}')

			print(f'estimated position is {index} with confidence {(posterior[index]*100):.4f}%:')
			

	barPlot(posterior)

	if do_print:
		print()
		print('final position is', robot.pos)
		index = np.argmax(posterior)
		print(f'Estimated position is {index} with confidence {(posterior[index]*100):.4f}%:')

if __name__=="__main__":
	train_filter(4, kernel=[.1, .8, .1], sensor_accuracy=.9,
         move_distance=4, do_print=True)