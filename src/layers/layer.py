from abc import ABC, abstractmethod, abstractproperty
from typing import Type
from src.activation_functions.activation_function import ActivationFunction
from src.optimizers.optimizer import Optimizer
import numpy as np


class Layer(ABC):
    @abstractmethod
    def __init__(self,activation_function: Type[ActivationFunction],optimizer: Type[Optimizer],**kwargs) -> None:
        ...

    @abstractmethod
    def forward_propagate(self,inputs: np.ndarray)-> np.ndarray:
        ...

    @abstractmethod
    def backward_propagate(self,prev_activated_values: np.ndarray,layer_gradient: np.ndarray)-> np.ndarray:
        ...

