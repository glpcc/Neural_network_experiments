import numpy as np
from src.activation_functions.activation_functions import *
'''
    HOW TO UPDATE WEIGHTS!!!!   
'''
loss_layer = np.array([[1],[2],[1]])
activated_out = np.array([[8,2],[1,4],[3,2]])
print(loss_layer.shape[0])
print(np.dot(loss_layer.T,activated_out)/loss_layer.shape[0])