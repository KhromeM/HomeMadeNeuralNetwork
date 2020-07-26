 
''' 
The goal is to make a neural network that works on data with 3 features and 
produces a prediction using linear regression  via gradient descent
'''

import numpy as np
from neurons import Neuron
# make the neuron class



class Layer():
	def __init__(self, neurons, previousLayer=None, nextLayer=None):
		self.neurons = neurons
		self.previousLayer = previousLayer
		self.nextLayer = nextLayer
		

	def generate(self):
		neurons = []
		for i in [1]*self.neurons:
			neurons.append(Neuron())
		self.neurons = neurons


l1 = Layer(3)
l1.generate()

print(l1.neurons)