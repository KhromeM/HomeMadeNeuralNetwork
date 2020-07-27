from layers import Layer
import numpy as np



class Model():

	def __init__(self,input):
		self.input = input
		self.layers = []

	def addLayer(self,neurons,activation):
		self.layers.append(Layer(neurons,activation))

	def backprop(self):
		self.layers.reverse()

		for i in self.layers:
			i.backprop()

		self.layers.reverse()



	def forward(self,counter):
		output = self.input
		for i in self.layers:
			output = i.forward(output)

		self.backprop()
		print(output, f'epoch: {counter}')
		return output

	def train(self,epochs):
		counter = 1
		while epochs >= counter:
			self.forward(counter)
			counter+=1



