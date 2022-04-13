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
