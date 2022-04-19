import random

from matplotlib import pyplot
from src.cost_funtions.cost_functions import CuadraticLoss,CrossEntropy
from src.neural_network import NeuralNetwork
from src.activation_functions.activation_functions import ReLu,softMax,sigmoid,no_op
import numpy as np

topology: list[int] = [2,1,2]
learning_rate: float = 1e-1
activation_functions = [no_op(),*[ReLu() for i in range(len(topology)-2)],softMax()]
cost_function = CrossEntropy()
batch_size = 2
test_size = 0
epochs = 1
net = NeuralNetwork(topology,activation_functions,cost_function,learning_rate)
errors = []
for i in range(epochs):
    if i%50 == 0:
        print(net.learning_rate)
        net.learning_rate /= 1.2
    inputs = np.zeros((batch_size,2))
    solutions = np.zeros((batch_size,2))
    for j in range(batch_size):
        inputs[j] = np.array([random.randint(0,100) for i in range(2)])
        if inputs[j][0] > inputs[j][1]:
            solutions[j] = np.array([0,1])
        else:
            solutions[j] = np.array([1,0])
    net.backward_propagation(inputs,solutions)

    inputs = np.zeros((test_size,2))
    solutions = np.zeros((test_size,2))
    for j in range(test_size):
        inputs[j] = np.array([random.randint(0,100) for i in range(2)])
        if inputs[j][0] > inputs[j][1]:
            solutions[j] = np.array([0,1])
        else:
            solutions[j] = np.array([1,0])
    predictions = net.feed_forward(inputs)
    errors.append(cost_function(solutions,predictions).mean())
    # print(f'Error:{cost_function(solutions,predictions).mean()}')
    random_prediction = predictions[20]
    random_solution = solutions[20]
    # print(f'Predicted:{random_prediction}, Actual:{random_solution}')
net.show_weights()
pyplot.plot(range(epochs),errors)
pyplot.show()