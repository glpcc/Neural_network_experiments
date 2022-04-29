from abc import ABC, abstractmethod
import numpy as np
from src.activation_functions.activation_function import ActivationFunction


class CostFunction(ABC):
    @abstractmethod
    def __call__(self,desired_values: np.ndarray,output_values: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def derv(self,desired_values: np.ndarray,output_values: np.ndarray) -> np.ndarray:
        ...




