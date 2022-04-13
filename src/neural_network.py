from random import randint, random
import numpy as np
from exceptions.exceptions import IncorrectInputSize
from activation_functions.activation_functions import *
from cost_funtions.cost_functions import *
from time import perf_counter
from matplotlib import pyplot as plt

class NeuralNetwork():

    def __init__(self,topology : list[int],activation_functions: list[ActivationFunction],cost_function : CostFunction,learning_rate :float) -> None:
        if len(activation_functions) == len(topology):
            self.__activation_functions = activation_functions
        else:
            raise Exception('Incorrect activation functions list size')
        self.__learning_rate = learning_rate
        self.__cost_function = cost_function
        self.__topology = topology
        self.__weights = [np.random.randn(j,topology[index]) for index,j in enumerate(topology[1:])] # Creates an array of 2D numpy arrays of size Current_layer_neurons*Prev_layer_neurons
        self.__biases = [np.random.randn(j) for j in topology[1:]]
        # Fill these arrays with dummy values and then will be used on backward propagation
        self.__activated_values = [np.zeros(j) for j in topology]
        self.__weighted_inputs = [np.zeros(j) for j in topology]


    def feed_forward(self,input_values: np.ndarray):
        '''
            PRE:
                The input must be a 2D array of inputs of size == input_neurons of the network
            POST:
                Returns an array of outputs for all the different values given
        '''
        if input_values.ndim != 2:
            raise IncorrectInputSize('The Input is not 2 dimensional numpy array')
        elif input_values.shape[1] != self.__topology[0]:
            raise IncorrectInputSize('The data of each input doesnt mach the input neurons')
    
        self.__weighted_inputs[0] = input_values
        self.__activated_values[0] = self.__activation_functions[0](input_values)
        for i in range(len(self.__topology)-1):
            self.__weighted_inputs[i+1] = np.dot(self.__activated_values[i],np.transpose(self.__weights[i]))+self.__biases[i]
            self.__activated_values[i+1] = self.__activation_functions[i+1](self.__weighted_inputs[i+1]) 

        return self.__activated_values[-1]


    def backward_propagation(self,batch: np.ndarray,desired_outputs: np.ndarray):
        if batch.ndim != 2:
            raise IncorrectInputSize('The batch is not 2 dimensional numpy array')
        elif desired_outputs.ndim != 2:
            raise IncorrectInputSize('The desired outputs is not 2 dimensional numpy array')

        outputs = self.feed_forward(batch)
        layer_errors = self.__cost_function.derv(desired_outputs,outputs) * self.__activation_functions[-1].derv(self.__weighted_inputs[-1])
        self._update_wb(layer_errors,len(self.__topology)-1)
        for layer_index in range(len(self.__topology)-2,0,-1):
            # Because the arent weights in the first layer I do not have to make layer_index + 1 (Im using backwards indexes)
            layer_errors = np.dot(layer_errors,self.__weights[layer_index]) * self.__activation_functions[layer_index].derv(self.__weighted_inputs[layer_index])
            self._update_wb(layer_errors,layer_index)


    def _update_wb(self,layer_errors: np.ndarray,layer_index: int):
        '''
            It updates weights and biases
            The layer error would be a 2D array of all the errors for all inputs in the batch 
        '''
        # The mean calculate the average of the errors in the batch for each neuron
        # The -1 is because there is no biases on the first layer
        self.__biases[layer_index-1] += layer_errors.mean(axis=0)*self.__learning_rate
        self.__weights[layer_index-1] += np.dot(layer_errors.T,self.__activated_values[layer_index-1])/layer_errors.shape[0]*self.__learning_rate

    def show_weights(self):
        for layer in self.__weights:
            print(layer)

    def show_biases(self):
        for layer in self.__biases:
            print(layer)

    def show_net_values(self):
        print(f'{self.__activated_values=}')
        print(f'{self.__weighted_inputs=}')

    @property
    def learning_rate(self):
        return self.__learning_rate
    
    @learning_rate.setter
    def learning_rate(self,rate):
        self.__learning_rate = rate

input_activation = no_op()
rest_activation = [no_op() for i in range(0)]
output_activation = no_op()
test = NeuralNetwork([2,1],[input_activation,*rest_activation,output_activation],CuadraticLoss(),1)
epochs = 1000
average_errors = []
for i in range(epochs):
    test.learning_rate = 0.001/(i+1)
    batch_size = 5
    input_values = np.empty((batch_size,2))
    solutions = np.empty((batch_size,1))
    for j in range(batch_size):
        n1 = random()*100
        n2 = random()*100
        input_values[j] = np.array([n1,n2])
        solutions[j] = np.array([n1+2*n2+2])
    test.backward_propagation(input_values,solutions)
    aux : float = 0
    test_size = 100
    for j in range(test_size):
        n1 = random()*100
        n2 = random()*100
        aux += n1+2*n2+2 - test.feed_forward(np.array([[n1,n2]]))[0][0]
    average_errors.append(aux/test_size)
plt.plot(list(range(epochs))[100:],average_errors[100:])
plt.show()