
import random
from matplotlib import pyplot
from src.layers.dense import Dense
from src.activation_functions.relu import ReLu
from src.activation_functions.softmax import SoftMax
from src.neural_network import NeuralNetwork
from src.optimizers.adam import Adam
from src.layers.layer import Layer
from src.cost_funtions.categorical_cross_entropy import CategoricalCrossEntropy
import numpy as np

np.set_printoptions(precision=2)


learning_rate: float =1e-3
inputs = 2
outputs = 2
layers: list[Layer] = [
    Dense(ReLu,Adam,inputs,2,learning_rate=learning_rate),
    Dense(SoftMax,Adam,2,outputs,learning_rate=learning_rate)
]

cost_function = CategoricalCrossEntropy()
batch_size = 5
test_size = 20
epochs = 300
net = NeuralNetwork(layers,cost_function)
errors = []
accuraccy = []
for i in range(epochs):
    batch_inputs = np.zeros((batch_size,inputs))
    batch_solutions = np.zeros((batch_size,outputs))
    for j in range(batch_size):
        batch_inputs[j] = np.array([random.randint(0,100),random.randint(0,100)])
        solutions = np.array([1,0]) if batch_inputs[j][0] >= batch_inputs[j][1] else np.array([0,1])
        batch_solutions[j] = np.array(solutions)

    net.backward_propagation(batch_inputs,batch_solutions)

    batch_inputs = np.zeros((test_size,inputs))
    batch_solutions = np.zeros((test_size,outputs))
    for j in range(test_size):
        batch_inputs[j] = np.array([random.randint(0,100),random.randint(0,100)])
        solutions = np.array([1,0]) if batch_inputs[j][0] >= batch_inputs[j][1] else np.array([0,1])
        batch_solutions[j] = np.array(solutions)
    predicted_results = net.feed_forward(batch_inputs)
    
    errors.append(cost_function(batch_solutions,predicted_results).mean())
    acc = np.where(np.argmax(predicted_results,axis=1)==np.argmax(batch_solutions,axis=1))[0].shape[0]/test_size
    accuraccy.append(acc)
    print(f'Epoch:{i},accuracy{round(acc*100,2)}% Error:{round(cost_function(batch_solutions,predicted_results).mean(),2)},Predicted:{predicted_results[5]},actual:{batch_solutions[5]}')


pyplot.plot(range(len(errors[0:])),errors[0:],label='Errors')
pyplot.plot(range(len(errors[0:])),accuraccy[0:],label='Accuracy')
pyplot.legend()
pyplot.xlabel('Epochs')
pyplot.show()