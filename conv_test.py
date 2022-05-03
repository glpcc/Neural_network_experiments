from src.layers.layer import Layer
from src.neural_network import NeuralNetwork
from src.layers.convolutional2D import Convolutional2D
from src.optimizers.adam import Adam
from src.activation_functions.relu import ReLu
from src.cost_funtions.categorical_cross_entropy import CategoricalCrossEntropy
from scipy.ndimage import convolve
import numpy as np
filter_shape = (2,2)
num_filters = 3
img_shape = (3,3)
cost_function = CategoricalCrossEntropy()
layers: list[Layer] = [
	Convolutional2D(ReLu,Adam,num_filters,filter_shape,img_shape)
]
net = NeuralNetwork(layers,cost_function)

test_imgs = np.arange(18).reshape(2,3,3)
print(layers[0].filters)
def func5(img,kernel):
	return convolve(img,np.rot90(kernel, 2),mode='valid')