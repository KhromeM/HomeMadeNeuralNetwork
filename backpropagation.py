import numpy as np 

'''
New Strategy:

Calculate the partial derivitive of the cost function for every neuron's weight and bias,
then update the weights and bias's by the derivative * -Alpha

Challenge: how to calculate derivitives
'''


def MSE(labels,outputs):
	return (labels-outputs)**2

def GD():
	pass