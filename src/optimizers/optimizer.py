from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):

    @abstractmethod
    def __init__(self,parameter_shape,**kwargs) -> None:
        ...

    @abstractmethod
    def calculate_parameter_change(self,gradient: np.ndarray) -> np.ndarray:
        ...
