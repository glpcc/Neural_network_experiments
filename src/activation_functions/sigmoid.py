import numpy as np
from src.activation_functions.activation_function import ActivationFunction


class Sigmoid(ActivationFunction):
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-array))

    def calculate_gradient(self, weighted_prev_errors: np.ndarray, weighted_inputs: np.ndarray) -> np.ndarray:
        return weighted_prev_errors*(self(weighted_inputs)*(1-self(weighted_inputs)))
