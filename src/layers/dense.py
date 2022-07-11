from src.activation_functions.activation_function import ActivationFunction
from src.exceptions.exceptions import IncorrectInputSize
from src.optimizers.optimizer import Optimizer
from src.layers.layer import Layer
from typing import Type
import numpy as np

class Dense(Layer):
    def __init__(self, activation_function: Type[ActivationFunction], optimizer: Type[Optimizer], num_inputs: int,num_outputs: int,**kwargs) -> None:
        self.__activation_function = activation_function()
        self.__weight_optimizer = optimizer((num_outputs,num_inputs),**kwargs)
        # Maybe add specific bias optimizer options later and even an specific optimizer
        self.__bias_optimizer = optimizer(num_outputs,**kwargs)
        # Initialize weights and biases
        self.__weights = (2*np.random.randn(num_outputs,num_inputs) - 1)/100  # The operations are to ensure values in the interval [-1,1]
        self.__biases = np.zeros(num_outputs)
        # Initialize the activated values and weighted inputs to zero for changing in the forward propagation
        self.__activated_values = np.zeros(num_outputs)
        self.__weighted_inputs = np.zeros(num_outputs)
        # Add num inputs and num outputs for shape checking in the class methods
        self.__num_inputs = num_inputs
        self.__num_outputs = num_outputs

    def forward_propagate(self, prev_activated_values: np.ndarray) -> np.ndarray:
        if prev_activated_values.shape[1] != self.__num_inputs:
            raise IncorrectInputSize('The Size of the inputs given didnt match the number of inputs of the layer')
        # Apply the weights and biases to the inputs
        self.__weighted_inputs = np.dot(prev_activated_values,self.__weights.T) +self.__biases
        # Apply the activation function to the weighted inputs
        self.__activated_values = self.__activation_function(self.__weighted_inputs)
        return self.__activated_values

    def backward_propagate(self, prev_activated_values: np.ndarray, prev_weighted_errors: np.ndarray) -> np.ndarray:
        layer_gradients = self.__activation_function.calculate_gradient(prev_weighted_errors,self.__weighted_inputs)
        # Optimize the weights and biases based on the gradient
        self.__optimize_weights_and_biases(prev_activated_values,layer_gradients)
        # Calculate the gradient for the next layer
        return np.dot(layer_gradients,self.__weights)


    def __optimize_weights_and_biases(self,prev_activated_values: np.ndarray, layer_gradients: np.ndarray) -> None:
        self.__weights += self.__weight_optimizer.calculate_parameter_change(np.dot(layer_gradients.T,prev_activated_values)) 
        self.__biases += self.__bias_optimizer.calculate_parameter_change(layer_gradients.sum(axis=0))
    
    @property
    def activated_values(self)-> np.ndarray:
        return self.__activated_values