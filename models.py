from layers import Layer
import numpy as np
from backpropagation import MSE


class Model():

	def __init__(self):
		self.layers = []

	def addLayer(self,neurons,activation):
		self.layers.append(Layer(neurons,activation))

	def backprop(self,labels):
		self.layers.reverse()
		# we reverse the layers to do backpropagation

		for i in self.layers:
			loss = MSE(labels, i.previous) # will calculate the average mean squared error
			print(loss, 'loss')
		self.layers.reverse()
		# we reverse it again to the normal oreintation before the next epoch



	def forward(self,inputs,counter,labels):
		outputs = inputs
		for i in self.layers:
			outputs = i.forward(outputs)

		self.backprop(labels)
		print(outputs, f'epoch: {counter}')
		return outputs

	def train(self,features,labels,epochs):
		#features will be called inputs from now on in code
		features = np.array(features)
		labels = np.array(labels)  # turn both in numpy arrays 
		counter = 1
		while epochs >= counter:
			self.forward(features,counter,labels)
			counter+=1



