from neurons import Neuron
from backpropagation import cost
import numpy as np 


class Layer():

	def __init__(self,neurons,activation):
		self.neurons = neurons
		self.activation = activation
		self.output = []

	def generate(self,inputs):
		neurons = []
		for i in [1]*self.neurons:
			neurons.append(Neuron(inputs,self.activation))
		self.neurons = neurons

	def backprop(self,labels):
		for n in self.neurons:
			#change = gradientDescent(n)
			n.weights +=.01
			n.bias +=.01



	def forward(self,inputs):
		self.output = []
		#intializes the neurons if self.neurons is an integer
		if type(self.neurons) == type(1):
			self.generate(inputs)
			
		#adds the output of each of the neurons in this layer to self.output
		for n in self.neurons:
			self.output.append(n.forward())

		return self.output
		




