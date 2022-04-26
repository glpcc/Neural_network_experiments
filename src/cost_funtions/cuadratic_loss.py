import numpy as np
from src.activation_functions.activation_function import ActivationFunction
from src.cost_funtions.cost_function import CostFunction


class CuadraticLoss(CostFunction):
    def __call__(self, desired_values: np.ndarray, output_values: np.ndarray) -> np.ndarray:
        return (np.square(desired_values - output_values)).mean(axis=1) # type:ignore

    def derv(self, desired_values: np.ndarray, output_values: np.ndarray, output_activation_function: ActivationFunction) -> np.ndarray:
        return output_activation_function.calculate_gradient(desired_values - output_values) # type:ignore
