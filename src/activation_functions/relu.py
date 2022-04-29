import numpy as np
from src.activation_functions.activation_function import ActivationFunction


class ReLu(ActivationFunction):
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return array*(array > 0)

    def calculate_gradient(self, weighted_prev_errors: np.ndarray, weighted_inputs: np.ndarray) -> np.ndarray:
        return weighted_prev_errors*(1*(weighted_inputs >= 0))
