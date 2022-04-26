from src.activation_functions.activation_function import ActivationFunction
from src.optimizers.optimizer import Optimizer
from src.layers.layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, activation_function: ActivationFunction, optimizer: Optimizer, num_input_values: int,num_output_values: int) -> None:
        self.__activation_function = activation_function
        self.__optmizer = optimizer
        self.__weights = 2*np.random.randn(num_output_values,num_input_values) - 1 # The operations are to ensure values in the interval [-1,1]
        self.__biases = np.zeros(num_output_values)
        self.__activated_values = np.zeros(num_output_values)
        self.__weighted_inputs = np.zeros(num_output_values)

    def forward_propagate(self, prev_activated_values: np.ndarray) -> np.ndarray:
        self.__weighted_inputs = np.dot(prev_activated_values,self.__weights.T) +self.__biases
        self.__activated_values = self.__activation_function(self.__weighted_inputs)
        return self.__activated_values

    def backward_propagate(self, prev_activated_values: np.ndarray, layer_gradients: np.ndarray) -> np.ndarray:
        self.__optimize_weights_and_biases(prev_activated_values,layer_gradients)
        return self.__activation_function.calculate_gradient(np.dot(layer_gradients,self.__weights),self.__weighted_inputs)


    def __optimize_weights_and_biases(self,prev_activated_values: np.ndarray, layer_gradients: np.ndarray) -> None:
        self.__weights += self.__optmizer.calculate_weight_change(np.dot(layer_gradients.T,prev_activated_values),0) #Quitar cero tras cambiar los optimizers
        self.__biases += self.__optmizer.calculate_bias_change(layer_gradients.sum(axis=0), 0) #Quitar cero tras cambiar los optimizers
        