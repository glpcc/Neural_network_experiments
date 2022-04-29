from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, array: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def calculate_gradient(self, weighted_prev_errors: np.ndarray, weighted_inputs: np.ndarray) -> np.ndarray:
        ...
