import random
from src.cost_funtions.cost_functions import CuadraticLoss,CrossEntropy
from src.neural_network import NeuralNetwork
from src.activation_functions.activation_functions import ReLu,softMax,sigmoid,no_op
import numpy as np


softmax = softMax()
cre = CrossEntropy()

predicted = np.array([[0.1,1],[0.3,0.9]])

test_predicted = np.array([[0.5,0.5]])
actual = np.array([[1,0],[0,1]])

print(test_predicted)
soft_predicted = softmax(predicted)
# error = cre(actual,soft_predicted)
error_grad = cre.derv(actual,soft_predicted)

print(error_grad)