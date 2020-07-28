 
''' 
The goal is to make a neural network from sratch that 
can compute in batches and use gradient descent
'''

import numpy as np
from models import Model 



X = np.array([.333,.45,-.34,-.254,-.85,.99,.75,.23])
y = [.33, -.45, .54]
model = Model()
model.addLayer(3,'relu')
model.addLayer(3,'relu')
#model.addLayer(1,'relu')
#model.addLayer(1,'sigmoid')
model.train(X,y,5)
