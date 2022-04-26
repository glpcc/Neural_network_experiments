import random

from matplotlib import pyplot
from src.cost_funtions.cost_functions import CuadraticLoss,CrossEntropy
from src.neural_network import NeuralNetwork
from src.activation_functions.activation_functions import ReLu,softMax,sigmoid,no_op
from src.optimizers.adam import Adam
import numpy as np

topology: list[int] = [2,2,2]
learning_rate: float = 1e-2
activation_functions = [no_op(),*[ReLu() for i in range(len(topology)-2)],softMax()]
cost_function = CrossEntropy()
batch_size = 10
test_size = 20
epochs = 500
optimizer = Adam(topology,learning_rate,0.9,0.999,1e-8)
net = NeuralNetwork(topology,activation_functions,cost_function,optimizer,learning_rate)
errors = []
for i in range(epochs):
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
    random_prediction = predictions[2]
    random_solution = solutions[2]
    print(f'Predicted:{random_prediction}, Actual:{random_solution}')

net.show_weights()
pyplot.plot(range(len(errors[:])),errors[:])
pyplot.show()