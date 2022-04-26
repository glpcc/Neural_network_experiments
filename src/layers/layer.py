from abc import ABC, abstractmethod
from src.activation_functions.activation_function import ActivationFunction
from src.optimizers.optimizer import Optimizer
import numpy as np


class Layer(ABC):
    @abstractmethod
    def __init__(self,activation_function: ActivationFunction,optimizer: Optimizer) -> None:
        ...

    @abstractmethod
    def forward_propagate(self,prev_activated_values: np.ndarray)-> np.ndarray:
        ...

    @abstractmethod
    def backward_propagate(self,prev_activated_values: np.ndarray,layer_gradient: np.ndarray)-> np.ndarray:
        ...