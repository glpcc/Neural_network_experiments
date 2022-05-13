from src.layers.layer import Layer
from src.neural_network import NeuralNetwork
from src.layers.convolutional2D import Convolutional2D
from src.optimizers.adam import Adam
from src.activation_functions.relu import ReLu
from src.activation_functions.no_op import No_op
from src.cost_funtions.categorical_cross_entropy import CategoricalCrossEntropy
from scipy.signal import convolve
import numpy as np
from matplotlib import pyplot
np.set_printoptions(precision=2)

filter_shape = (2,2)
num_filters = 2
img_shape = (3,3)
cost_function = CategoricalCrossEntropy()
layers: list[Convolutional2D] = [
	Convolutional2D(No_op,Adam,num_filters,filter_shape,img_shape,learning_rate=1e-2)
]
net = NeuralNetwork(layers,cost_function)

test_imgs = np.arange(9).reshape(1,3,3)
objective_kernels = np.random.randn(2,2,2)
print(objective_kernels)
batches = 1000
errors = []
errors2 = []
for i in range(batches):
	test_imgs = np.random.randn(1,3,3)*10
	expected_outs = np.empty((2,2,2))
	for i in range(len(objective_kernels)):
		expected_outs[i] = convolve(test_imgs[0],np.rot90(objective_kernels[i],2),mode='valid')
	outputs = layers[0].forward_propagate(test_imgs)
	error = expected_outs.reshape(1,8) - outputs
	errors.append(error.sum())
	errors2.append((np.square(expected_outs.reshape(1,8) - outputs)).mean(axis=1))
	layers[0].backward_propagate(test_imgs,error)

print(layers[0].filters)
pyplot.plot(list(range(batches)),errors)
pyplot.show()
