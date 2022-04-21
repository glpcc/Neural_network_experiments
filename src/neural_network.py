import numpy as np
from src.exceptions.exceptions import IncorrectInputSize
from src.activation_functions.activation_functions import ActivationFunction
from src.cost_funtions.cost_functions import CostFunction

class NeuralNetwork():

    def __init__(self,topology : list[int],activation_functions: list[ActivationFunction],cost_function : CostFunction,learning_rate :float) -> None:
        if len(activation_functions) == len(topology):
            self.__activation_functions = activation_functions
        else:
            raise Exception('Incorrect activation functions list size')
        self.__learning_rate = learning_rate
        self.__cost_function = cost_function
        self.__topology = topology
        # Starts the weights with random values for diferent gradients
        self.__weights = [np.random.rand(j,topology[index])/10 for index,j in enumerate(topology[1:])] # Creates an array of 2D numpy arrays of size Current_layer_neurons*Prev_layer_neurons
        self.__biases = [np.zeros(j)/10 for j in topology[1:]]
        # Fill these arrays with dummy values and then will be used on backward propagation
        self.__activated_values = [np.zeros(j) for j in topology]
        self.__weighted_inputs = [np.zeros(j) for j in topology]


    def feed_forward(self,input_values: np.ndarray):
        '''
            PRE:
                The input must be a 2D array of inputs of size == input_neurons of the network
            POST:
                Returns an array of outputs for all the different inputs given
        '''
        # Check for inputs shapes
        if input_values.ndim != 2:
            raise IncorrectInputSize('The Input is not 2 dimensional numpy array')
        elif input_values.shape[1] != self.__topology[0]:
            raise IncorrectInputSize('The data of each input doesnt match the input neurons')
    
        self.__weighted_inputs[0] = input_values
        self.__activated_values[0] = self.__activation_functions[0](input_values)
        for i in range(len(self.__topology)-1):
            self.__weighted_inputs[i+1] = np.dot(self.__activated_values[i],np.transpose(self.__weights[i]))+self.__biases[i]
            self.__activated_values[i+1] = self.__activation_functions[i+1](self.__weighted_inputs[i+1]) 

        return self.__activated_values[-1]

    def backward_propagation(self,batch: np.ndarray,desired_outputs: np.ndarray):
        # Check for inputs shapes
        if batch.ndim != 2:
            raise IncorrectInputSize(f'The batch is not 2 dimensional numpy array the given shape was {batch.shape}')
        elif desired_outputs.ndim != 2:
            raise IncorrectInputSize(f'The desired outputs is not 2 dimensional numpy the given shape was {desired_outputs.shape}')
        # Forward propagate the batch
        outputs = self.feed_forward(batch)
        # Calculate the last layer errors
        layer_errors = self.__cost_function.derv(desired_outputs,outputs,self.__activation_functions[-1])
        # Update the last layer weights and biases and the backpropagate the errors to the rest of the layers
        self._update_wb(layer_errors,len(self.__topology)-1)
        for layer_index in range(len(self.__topology)-2,0,-1):
            layer_errors = self.__activation_functions[layer_index].calculate_errors(np.dot(layer_errors,self.__weights[layer_index]),self.__weighted_inputs[layer_index])
            self._update_wb(layer_errors,layer_index)

    def _update_wb(self,layer_errors: np.ndarray,layer_index: int):
        '''
            It updates weights and biases
            The layer error would be a 2D array of all the errors for all inputs in the batch 
        '''
        # I changed the biases by the mean of the batches errors
        self.__biases[layer_index-1] += layer_errors.mean(axis=0)*self.__learning_rate
        # I multiply the errors with the previous layer values and then take the average to change the weights
        self.__weights[layer_index-1] += np.dot(layer_errors.T,self.__activated_values[layer_index-1])/layer_errors.shape[0]*self.__learning_rate

    def show_weights(self):
        for layer in self.__weights:
            print(layer)

    def show_biases(self):
        for layer in self.__biases:
            print(layer)

    def show_net_values(self):
        print(f'{self.__activated_values=}')
        print(f'{self.__weighted_inputs=}')

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self,rate):
        self.__learning_rate = rate
