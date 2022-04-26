import numpy as np
from src.activation_functions.activation_function import ActivationFunction


class Tanh(ActivationFunction):
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return np.tanh(array)

    def calculate_gradient(self, weighted_prev_errors: np.ndarray, weighted_inputs: np.ndarray) -> np.ndarray:
        b = np.tanh(weighted_inputs)
        return weighted_prev_errors*(1 - b*b)
