from typing import Callable,Optional
import numpy as np
from exceptions.exceptions import IncorrectInputSize
from time import perf_counter
import math

def no_op(a: np.ndarray):
    return a

class NeuralNetwork():
    def __init__(self,topology : list[int],activation_functions: Optional[list[Callable]] = None) -> None:
        if activation_functions == None:
            self.__activation_functions = [no_op for i in range(1,len(topology))]
        else:
            if len(activation_functions) == len(topology) -1:
                self.__activation_functions = activation_functions
            else:
                raise Exception('Incorrect activation functions list size')
                
        self.__topology = topology
        self.__weights = [np.random.randn(j,topology[index]) for index,j in enumerate(topology[1:])]
        self.__biases = [np.random.randn(j,topology[index]) for index,j in enumerate(topology[1:])]
        self.__values = [np.zeros(j) for j in topology]

    def feed_forward(self,input_values: np.ndarray):
        if len(input_values) == self.__topology[0]:
            self.__values[0] = input_values

            for i in range(len(self.__topology)-1):
                self.__values[i+1] = self.__activation_functions[i](np.sum(self.__values[i]*self.__weights[i]+self.__biases[i],axis=1))

            return self.__values[-1]
        else:
            raise IncorrectInputSize


    def show_weights(self):
        for layer in self.__weights:
            print(layer)

    def show_biases(self):
        for layer in self.__biases:
            print(layer)

    def show_net_values(self):
        print(self.__values)

test = NeuralNetwork([2,2],[no_op for i in range(1)])
a1 = perf_counter()
print(test.feed_forward(np.array([2,2])))
a2 = perf_counter()
test.show_weights()
test.show_biases()
test.show_net_values()
print(a2-a1)