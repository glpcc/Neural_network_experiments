from abc import ABC, abstractmethod
import numpy as np

class CostFunction(ABC):
    @abstractmethod
    def __call__(self,desired_values: np.ndarray,output_values: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def derv(self,desired_values: np.ndarray,output_values: np.ndarray) -> np.ndarray:
        ...


class CuadraticLoss(CostFunction):
    def __call__(self,desired_values: np.ndarray,output_values: np.ndarray) -> np.ndarray:
        return (np.square(desired_values - output_values)).mean(axis=1) # type:ignore

    def derv(self,desired_values: np.ndarray,output_values: np.ndarray) -> np.ndarray:
        return desired_values - output_values  # type:ignore

class CrossEntropy(CostFunction):
    '''
        This is supossed to be used when softmax is used as a the last layer activation
    '''
    def __call__(self,desired_values: np.ndarray,output_values: np.ndarray) -> np.ndarray:
        output_cliped = np.clip(output_values,1e-9,1-1e-9)
        return -np.sum(desired_values*np.log(output_cliped),axis=1)/(output_cliped.shape[1]) # type:ignore

    def derv(self,desired_values: np.ndarray,output_values: np.ndarray) -> np.ndarray:
        m = desired_values.shape[0]
        grad = output_values
        grad[range(m),np.argmax(desired_values,axis=1)] -= 1
        grad = grad/m
        return -grad

