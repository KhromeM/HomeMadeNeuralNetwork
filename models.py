from layers import Layer
import numpy as np



class Model():

	def __init__(self):
		self.layers = []

	def addLayer(self,neurons,activation):
		self.layers.append(Layer(neurons,activation))

	def backprop(self,labels,outputs):
		self.layers.reverse()
		outputs.reverse()
		outputs = np.array(outputs)

		for i in self.layers:
			i.backprop(labels,outputs)

		self.layers.reverse()



	def forward(self,inputs,counter,labels):
		outputs = inputs
		for i in self.layers:
			outputs = i.forward(outputs)

		self.backprop(labels,outputs)
		print(output, f'epoch: {counter}')
		return output

	def train(self,features,labels,epochs):
		#features will be called inputs from now on in code
		counter = 1
		while epochs >= counter:
			self.forward(features,counter,labels)
			counter+=1



