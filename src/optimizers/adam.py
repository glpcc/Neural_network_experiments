from src.optimizers.optimizers import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(self,topology: list[int],learning_rate: float,beta1: float,beta2: float,epsilon: float) -> None:
        self.__weights_m = [np.zeros((j,topology[index])) for index,j in enumerate(topology[1:])]
        self.__weights_v = [np.zeros((j,topology[index])) for index,j in enumerate(topology[1:])]
        self.__biases_m = [np.zeros(j) for j in topology[1:]]
        self.__biases_v = [np.zeros(j) for j in topology[1:]]
        self.__beta1 = beta1
        self.__beta2 = beta2
        self.__epsilon = epsilon
        self.__learning_rate = learning_rate
        self.epoch = 2

    def calculate_weight_change(self,average_gradient: np.ndarray,layer: int)-> np.ndarray:
        self.epoch += 1
        self.__weights_m[layer] = self.__beta1*self.__weights_m[layer] + (1-self.__beta1)*average_gradient
        self.__weights_v[layer] = self.__beta2*self.__weights_v[layer] + (1-self.__beta2)*np.square(average_gradient)
        return self.__learning_rate*(self.__weights_m[layer]/(np.sqrt(self.__weights_v[layer])+self.__epsilon))

    def calculate_bias_change(self,average_gradient: np.ndarray,layer: int)-> np.ndarray:
        self.__biases_m[layer] = self.__beta1*self.__biases_m[layer] + (1-self.__beta1)*average_gradient
        self.__biases_v[layer] = self.__beta2*self.__biases_v[layer] + (1-self.__beta2)*np.square(average_gradient)
        return self.__learning_rate*(self.__biases_m[layer]/(np.sqrt(self.__biases_v[layer])+self.__epsilon))