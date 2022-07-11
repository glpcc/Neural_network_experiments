import numpy as np
from src.cost_funtions.cost_function import CostFunction
from src.layers.layer import Layer
class NeuralNetwork():

    def __init__(self,layers: list[Layer],cost_function : CostFunction) -> None:
        self.__cost_function: CostFunction = cost_function
        self.__layers: list[Layer] = layers


    def feed_forward(self,input_values: np.ndarray):
        inputs = input_values
        for layer in self.__layers:
            inputs = layer.forward_propagate(inputs)
        return inputs


    def backward_propagation(self,batch: np.ndarray,desired_outputs: np.ndarray):
        output = self.feed_forward(batch)
        error = self.__cost_function.derv(desired_outputs,output)
        # print(f'ERR:{error.mean()}')
        for i in range(len(self.__layers)-1,0,-1):
            error = self.__layers[i].backward_propagate(self.__layers[i-1].activated_values,error)
        self.__layers[0].backward_propagate(batch,error)
    
    def test(self,batch: np.ndarray, desired_outputs: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        output = self.feed_forward(batch)
        error = self.__cost_function.derv(desired_outputs,output)
        return error, output
