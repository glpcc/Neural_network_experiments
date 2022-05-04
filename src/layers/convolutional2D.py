from src.layers.layer import Layer
from src.activation_functions.activation_function import ActivationFunction
from src.optimizers.optimizer import Optimizer
from typing import Optional, Type
from scipy.signal import convolve
import numpy as np


class Convolutional2D(Layer):
    '''
        This is a 2D concolutional layer
    '''

    def __init__(self, activation_function: Type[ActivationFunction], optimizer: Type[Optimizer],num_filters: int,filter_shape: tuple[int,...],input_img_shape: tuple[int,...], **kwargs) -> None:
        num_outputs = kwargs.get('num_outputs',None)
        self.__activation_function = activation_function()
        self.__filter_optimizer = optimizer((num_filters,*filter_shape),**kwargs)
        self.__num_filters = num_filters
        self.__filter_shape = filter_shape
        # Research how to initialize if not working
        self.__filters = 2*np.random.randn(num_filters,*filter_shape) -1
        #
        # Add bias later
        #
        self.__input_img_shape = input_img_shape
        # for convenience
        self.__output_img_shape = (input_img_shape[0]-filter_shape[0]+1,input_img_shape[1]-filter_shape[1]+1)
        # check if number of outputs where given
        if num_outputs:
            if num_outputs != (self.__output_img_shape[0]*self.__output_img_shape[1])*num_filters:
                raise ValueError('The num of outputs doesnt correspond to the input img shape and the filter shape')
            else:
                self.__num_outputs = num_outputs
        else:
            self.__num_outputs = (self.__output_img_shape[0]*self.__output_img_shape[1])*num_filters
        # Initialize this array for storing the results of each epoch for use in backprop
        self.__weighted_inputs: np.ndarray 
        self.__activated_values: np.ndarray
    def forward_propagate(self, prev_activated_values: np.ndarray) -> np.ndarray:
        '''
            Im using scipy convolve because of the massive performance improvement(>10x) compared to python loops and better performance for bigger filter
            Further improvement posibilities:
            - Use convolve2d if filters are small
            - Use cupyx for gpu acceleration
        '''
        # Adapt input data
        input_data = self._adapt_input_shape(prev_activated_values)
        outputs = np.empty((prev_activated_values.shape[0],self.__num_filters,*self.__output_img_shape))
        
        # Maybe Optimize this loops later with  cupy or numba
        for i,image in enumerate(input_data):
            for j,filter in enumerate(self.__filters):
                outputs[i,j] = convolve(image,np.rot90(filter,2),mode='valid')

        self.__weighted_inputs = outputs.reshape(prev_activated_values.shape[0],self.__num_filters*self.__output_img_shape[0]*self.__output_img_shape[1])
        self.__activated_values =  self.__activation_function(self.__weighted_inputs)
        return self.__activated_values

    def backward_propagate(self, prev_activated_values: np.ndarray, prev_weighted_errors: np.ndarray) -> np.ndarray:
        '''
            Filter_gradient = conv(input,gradient)
            input_gradient = full_conv(filter,gradient)
        '''
        # Adapt input data
        input_data = self._adapt_input_shape(prev_activated_values)
        # Calculate the gradient
        layer_gradients = self.__activation_function.calculate_gradient(prev_weighted_errors,self.__weighted_inputs)
        layer_gradients = self._adapt_gradient_shape(layer_gradients)
        # Calculate the filter gradient and input_gradient
        input_gradient = np.empty(self.__input_img_shape)
        filter_gradients = np.empty((input_data.shape[0],self.__num_filters,*self.__filter_shape))
        for i in range(len(input_data)):
            for j in range(len(layer_gradients[i])):
                input_gradient += convolve(self.filters[j],layer_gradients[i][j],mode='full')
                filter_gradients[i][j] = convolve(input_data[i],layer_gradients[i][j],mode='valid')
        self._update_filters(filter_gradients.sum(axis=0))
        return input_gradient

    def _update_filters(self,filter_gradient: np.ndarray)-> None:
        # USE OPTIMIZER!!!!!
        if filter_gradient.shape != (self.__num_filters,*self.__filter_shape):
            raise ValueError('The filter gadient is not in the correct shape')
        else:
            filter_delta = self.__filter_optimizer.calculate_parameter_change(filter_gradient)
            for fltr,delta in zip(self.filters,filter_delta):
                fltr += delta


    def _adapt_input_shape(self,input: np.ndarray)-> np.ndarray:
        if input.shape[1:] == self.__input_img_shape:
            input_data = input
        elif input.shape[1] == self.__input_img_shape[0]*self.__input_img_shape[1]:
            input_data = input.reshape(input.shape[0],*self.__input_img_shape) 
        else:
            raise ValueError('Incorrect input size')

        return input_data
    
    def _adapt_gradient_shape(self,gradient: np.ndarray)-> np.ndarray:
        if gradient.shape[1] == self.__num_filters*self.__output_img_shape[0]*self.__output_img_shape[1]:
            new_gradient = gradient.reshape((gradient.shape[0],self.__num_filters,*self.__output_img_shape))
        else:
            print(gradient)
            raise ValueError(f'The gradient has the wrong dimensions {gradient.shape} instead of {(gradient.shape[0],self.__num_filters*self.__output_img_shape[0]*self.__output_img_shape[1])}')
        return new_gradient
    
    
    @property
    def activated_values(self)-> np.ndarray:
        return self.__activated_values


    # Just for testing REMOVE later
    @property
    def filters(self):
        return self.__filters
