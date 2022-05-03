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

    def __init__(self, activation_function: Type[ActivationFunction], optimizer: Type[Optimizer], /
                 num_filters: int,filter_shape: tuple[int,...],input_img_shape: tuple[int,...], **kwargs) -> None:
        num_outputs = kwargs.get('num_outputs',None)
        self.__activation_function = activation_function()
        self.__filter_optimizer = optimizer((num_filters,*filter_shape),**kwargs)
        self.__num_filters = num_filters
        self.__filter_shape = filter_shape
        # Research how to initialize if not working
        self.__filters = 2*np.random.randn(num_filters,*filter_shape) - 1
        self.__input_img_shape = input_img_shape
        # for convenience
        self.__output_img_shape = (input_img_shape[0]-filter_shape[0]+1,input_img_shape[1]-filter_shape[1]+1)
        if num_outputs:
            if num_outputs != (self.__output_img_shape[0]*self.__output_img_shape[1])*num_filters:
                raise ValueError('The num of outputs doesnt correspond to the input img shape and the filter shape')
            else:
                self.__num_outputs = num_outputs
        else:
            self.__num_outputs = (self.__output_img_shape[0]*self.__output_img_shape[1])*num_filters

    def forward_propagate(self, prev_activated_values: np.ndarray) -> np.ndarray:
        # Im using scipy convolve because of the massive performance improvement(>10x) compared to python loops and better performance for bigger filter
        # Further improvement posibilities:
        #   - Use convolve2d if filters are small
        #   - Use cupyx for gpu acceleration
        if prev_activated_values.shape[1:] == self.__input_img_shape:
            input_data = prev_activated_values
        elif prev_activated_values.shape[1] == self.__input_img_shape[0]*self.__input_img_shape[1]:
            input_data = prev_activated_values.reshape(prev_activated_values.shape[0],*self.__input_img_shape) 
        else:
            raise ValueError('Incorrect input size')
        
        outputs = np.empty((prev_activated_values.shape[0],self.__num_filters,*self.__output_img_shape))
        # Maybe Optimize this loops later with  cupy or numba
        for i,image in enumerate(input_data):
            for j,filter in enumerate(self.__filters):
                outputs[i,j] = convolve(image,np.rot90(filter,2),mode='valid')

        return self.__activation_function(outputs.reshape(prev_activated_values.shape[0],self.__num_filters*self.__output_img_shape[0]*self.__output_img_shape[1]))

    def backward_propagate(self, prev_activated_values: np.ndarray, layer_gradient: np.ndarray) -> np.ndarray:
        ...


    # Just for testing REMOVE later
    @property
    def filters(self):
        return self.__filters