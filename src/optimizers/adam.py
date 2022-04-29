from src.optimizers.optimizer import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(self,parameter_shape: tuple[int,...],**kwargs) -> None:
        self.__parameter_m = np.zeros(parameter_shape)
        self.__parameter_v = np.zeros(parameter_shape)
        self.beta1 = kwargs.get('beta1',0.9)
        self.beta2 = kwargs.get('beta2',0.999)
        self.epsilon = kwargs.get('epsilon',1e-8)
        self.learning_rate = kwargs.get('learning_rate',1e-2)

    def calculate_parameter_change(self,gradient: np.ndarray)-> np.ndarray:
        self.__parameter_m = self.beta1*self.__parameter_m + (1-self.beta1)*gradient
        self.__parameter_v = self.beta2*self.__parameter_v + (1-self.beta2)*np.square(gradient)
        return self.learning_rate*(self.__parameter_m/(np.sqrt(self.__parameter_v)+self.epsilon))
