from src.activation_functions.softmax import SoftMax
from src.cost_funtions.cost_function import CostFunction
from src.activation_functions.activation_function import ActivationFunction
import numpy as np


class CategoricalCrossEntropy(CostFunction):
    '''
        This is supossed to be used when softmax is used as a the last layer activation
    '''

    def __call__(self, desired_values: np.ndarray, output_values: np.ndarray) -> np.ndarray:
        output_cliped = np.clip(output_values, 1e-9, 1-1e-9)
        return -np.sum(desired_values*np.log(output_cliped), axis=1)/(output_cliped.shape[1])

    def derv(self, desired_values: np.ndarray, output_values: np.ndarray) -> np.ndarray:
        output_cliped = np.clip(output_values,1e-7,1-1e-7)
        return desired_values/output_cliped