from typing import Optional
import numpy as np
from exceptions.exceptions import IncorrectInputSize
from activation_functions.activation_functions import *
from time import perf_counter

class NeuralNetwork():
    def __init__(self,topology : list[int],activation_functions: Optional[list[ActivationFunction]] = None) -> None:
        if activation_functions == None:
            self.__activation_functions = [no_op() for i in range(len(topology))]
        else:
            if len(activation_functions) == len(topology):
                self.__activation_functions = activation_functions
            else:
                raise Exception('Incorrect activation functions list size')
                
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


    def backward_propagation(self,batch: np.ndarray):
        if batch.ndim != 2:
            raise IncorrectInputSize('The batch is not 2 dimensional numpy array')

        
        

    def show_weights(self):
        for layer in self.__weights:
            print(layer)

    def show_biases(self):
        for layer in self.__biases:
            print(layer)

    def show_net_values(self):
        print(f'{self.__activated_values=}')
        print(f'{self.__weighted_inputs=}')

input_activation = no_op()
rest_activation = [tanh() for i in range(2)]

test = NeuralNetwork([2,3,4],[input_activation,*rest_activation])
print(test.feed_forward(np.array([[1,2],[2,1]])))
test.show_net_values()
