from abc import ABC, abstractmethod
import numpy as np

class CostFunction(ABC):
    @abstractmethod
    def __call__(self,array: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def derv(self,array: np.ndarray) -> np.ndarray:
        ...


class CuadraticLoss(ABC):
    def __call__(self,desired_values: np.ndarray,output_values: np.ndarray) -> np.ndarray:
        return (np.square(desired_values - output_values)).mean(axis=1) # type:ignore

    def derv(self,desired_values: np.ndarray,output_values: np.ndarray) -> np.ndarray:
        return output_values - desired_values  # type:ignore

a = CuadraticLoss()

print(a.derv(np.array([[1,1,1],[2,3,4]]),np.array([[1,2,1],[7,6,4]])))