 
''' 
The goal is to make a neural network that works on data with 3 features and 
produces a prediction using linear regression  via gradient descent
'''
import numpy as np
from models import Model 



inpt = np.array([.333,.45,-.34,-.254,-.85,.99,.75,.23])
model = Model(inpt)
model.addLayer(2,'relu')
model.addLayer(1,'relu')
model.addLayer(1,'relu')
model.addLayer(1,'sigmoid')
model.forward()
