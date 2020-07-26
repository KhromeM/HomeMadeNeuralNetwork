from layers import Layer
import numpy as np



class Model():

	def __init__(self,input):
		self.input = input
		self.layers = []

	def addLayer(self,neurons,activation):
		self.layers.append(Layer(neurons,activation))

	def forward(self):
		for i in self.layers:
			self.input = i.forward(self.input)

		print(self.input)
		return self.input




