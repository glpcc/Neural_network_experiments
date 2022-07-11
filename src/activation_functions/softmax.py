import numpy as np
from src.activation_functions.activation_function import ActivationFunction


class SoftMax(ActivationFunction):
    def __call__(self, array: np.ndarray) -> np.ndarray:
        maxs = array.max(axis=1, keepdims=True)
        exps = np.exp(array - maxs)
        sums = np.sum(exps, axis=1, keepdims=True)
        return exps / sums

    def calculate_gradient(self, weighted_prev_errors: np.ndarray, weighted_inputs: np.ndarray) -> np.ndarray:
        errors = np.zeros(weighted_prev_errors.shape)
        softmax = self(weighted_inputs)
        for i in range(weighted_prev_errors.shape[0]):
            temp = np.reshape(softmax[i], (1, -1))
            errors[i] = weighted_prev_errors[i].reshape(
                (1, -1))@(temp * np.identity(temp.size) - temp.T @ temp)
        return errors
