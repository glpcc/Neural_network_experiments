from src.layers.layer import Layer
from src.neural_network import NeuralNetwork
from src.layers.convolutional2D import Convolutional2D
from src.optimizers.adam import Adam
from src.activation_functions.relu import ReLu
from src.activation_functions.no_op import No_op
from src.cost_funtions.categorical_cross_entropy import CategoricalCrossEntropy
from scipy.signal import convolve
import numpy as np
filter_shape = (2,2)
num_filters = 3
img_shape = (3,3)
cost_function = CategoricalCrossEntropy()
layers: list[Layer] = [
	Convolutional2D(No_op,Adam,num_filters,filter_shape,img_shape)
]
net = NeuralNetwork(layers,cost_function)

test_imgs = np.arange(18).reshape(2,3,3)
print(layers[0].filters)

def func5(imgn,ker):
	return convolve(imgn,np.rot90(ker, 2),mode='valid')


print(net.feed_forward(test_imgs))
print(func5(test_imgs[0],layers[0].filters[0]))