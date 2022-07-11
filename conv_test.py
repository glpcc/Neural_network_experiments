from src.cost_funtions.cost_function import CostFunction
from src.layers.layer import Layer
from src.neural_network import NeuralNetwork
from src.layers.convolutional2D import Convolutional2D
from src.optimizers.adam import Adam
from src.activation_functions.relu import ReLu
from src.activation_functions.no_op import No_op
from src.cost_funtions.cuadratic_loss import CuadraticLoss
from scipy.signal import convolve
import numpy as np
from matplotlib import pyplot
np.set_printoptions(precision=2)

filter_shape1 = (4,4)
filter_shape2 = (2,2)
num_filters = 1
img_shape = (9,9)
output1_img_shape = (img_shape[0]-filter_shape1[0]+1,img_shape[1]-filter_shape1[1]+1)
cost_function = CuadraticLoss()
layers: list[Convolutional2D] = [
	Convolutional2D(No_op,Adam,num_filters,filter_shape1,img_shape,learning_rate=1e-1),
	Convolutional2D(No_op,Adam,num_filters,filter_shape2,output1_img_shape,learning_rate=1e-1)
]
net = NeuralNetwork(layers,cost_function)

expected_kernel1 = np.random.randn(*filter_shape1)/2
expected_kernel2 = np.random.randn(*filter_shape2)/2
print(expected_kernel1,expected_kernel2)
batches = 1000
errors = []
for i in range(batches):
	random_image: np.ndarray = np.random.randn(1,*img_shape)*10
	expected_out1 = convolve(random_image[0],np.rot90(expected_kernel1,2),mode='valid')
	expected_out2 = convolve(expected_out1,np.rot90(expected_kernel2,2),mode='valid')
	net_output = net.feed_forward(random_image)
	error = cost_function(expected_out2.reshape(1,25),net_output)
	errors.append(error.sum())
	net.backward_propagation(random_image,expected_out2.reshape(1,25))
	if i == batches-1:
		print(net_output)
		print(expected_out2)

print(layers[0].filters)
print(layers[1].filters)
pyplot.plot(list(range(batches)),errors)
pyplot.show()

