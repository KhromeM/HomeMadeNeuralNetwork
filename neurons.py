import numpy as np
import math


class Neuron():
	def __init__(self,inputs,activation):
		self.inputs = inputs #a vector received as the output of the previous layer
		self.weights = [] # a vector the same length as the input, first intialized as 0
		self.activation = activation
		self.output = 0 
		self.bias = np.random.randn((1))[0]

	def relu(self,z):
		return max(0,z)

	def sigmoid(self,z):
		if z>500:
			return 1
		return ((math.e)**z)/((math.e)**z +1)

	def activate(self):

		if self.activation == 'relu':
			self.activation = self.relu

		if self.activation =='sigmoid':
			self.activation= self.sigmoid

	def forward(self):
		#if there are no weights, generate weights
		if len(self.weights) == 0 :
			self.weights = np.random.randn((len(self.inputs)))
		# set the activation function
		self.activate()
		# dot product the inputs and weights then add bias
		#print(self.inputs)
		#print(self.bias)
		print(self.weights)
		z = np.dot(self.weights,self.inputs) + self.bias
		#print(z)
		# use the activation function to calculate the output
		self.output = self.activation(z)
		return(self.output)







		

