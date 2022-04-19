import random
from src.cost_funtions.cost_functions import CuadraticLoss,CrossEntropy
from src.neural_network import NeuralNetwork
from src.activation_functions.activation_functions import ReLu,softMax,sigmoid,no_op
import numpy as np


flower_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica':2
} 

print(list(flower_map.keys())[0])