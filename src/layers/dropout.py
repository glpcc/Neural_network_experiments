from src.activation_functions.activation_function import ActivationFunction
from src.exceptions.exceptions import IncorrectInputSize
from src.optimizers.optimizer import Optimizer
from src.layers.layer import Layer
from typing import Type
import numpy as np

class Dropout(Layer):
    def __init__(self,probability: float, num_inputs: int) -> None:
        self.__probability = probability
        self.__num_inputs = num_inputs
        self.__droped_nodes = np.ones(num_inputs)
        self.random_state = np.random.default_rng()

    def forward_propagate(self, prev_activated_values: np.ndarray,learning: bool = True) -> np.ndarray:
        if learning:
            self.__droped_nodes = np.random.Generator.binomial(self.random_state,1, self.__probability, self.__num_inputs)
            return self.__droped_nodes * prev_activated_values
        else:
            return prev_activated_values

    def backward_propagate(self, prev_activated_values: np.ndarray, prev_weighted_errors: np.ndarray) -> np.ndarray:
        return self.__droped_nodes * prev_weighted_errors

