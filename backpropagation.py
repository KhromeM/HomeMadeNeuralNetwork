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
First lets implement backpropagation fo a simple +1 to every weight and bias
'''
def cost(y,weights,inputs):
	gr