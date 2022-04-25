from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):
    @abstractmethod
    def calculate_weight_change(self,average_gradient: np.ndarray,layer: int) -> np.ndarray:
        ...

    @abstractmethod
    def calculate_bias_change(self,average_gradient: np.ndarray,layer: int)-> np.ndarray:
        ...