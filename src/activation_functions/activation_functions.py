from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self,array: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def derv(self,array: np.ndarray) -> np.ndarray:
        ...


class no_op(ActivationFunction):
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return array
    
    def derv(self, array: np.ndarray) -> np.ndarray:
        return np.array([1])


class tanh(ActivationFunction):
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return np.tanh(array)
    
    def derv(self, array: np.ndarray) -> np.ndarray:
        b = np.tanh(array)
        return 1 - b*b
