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




inpt = np.array([.333,.45,-.34,-.254,-.85,.99,.75,.23])
model = Model(inpt)
model.addLayer(2,'relu')
model.addLayer(1,'relu')
model.addLayer(1,'relu')
model.addLayer(1,'sigmoid')
model.forward()

#print(model.layers[-1].neurons)