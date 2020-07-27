import numpy as np 

'''
How will this work? (lol I dont know)

A = learning rate

Step 1) We need a Cost function!, let it be MSE for linear regression
y = prediction
a = actual
MSE = (y-a)**2
lets break a into its components:
a = activation(inputs (dot) weights + bias) //the activation function can be relu or sigmoid (what in the wolrd is the derivative for the relu?!?!)


Step 2)  start at the last layer, iterate over over every neuron, do :
		
			calculate the gradient for the weights, store it as G
			//calculate the partial derivitive of the activation function, store it as A
			//calculate the partial derivitive of the bias, store it as B

step 3) 
		weights -= G*a
					 
'''

'''
Strat for backpropagation:
make every layer remeber its most recent output, then use that in GD
call backpropagation from the Model class to keep the other code clean
'''
def MSE(labels,outputs):
	return np.sum((labels-outputs)**2)

def GD():
	pass