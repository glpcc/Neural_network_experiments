
import random
from matplotlib import pyplot
from src.cost_funtions.cost_function import CuadraticLoss,CrossEntropy
from src.neural_network import NeuralNetwork
from src.activation_functions.activation_function import ReLu,softMax,sigmoid,no_op
from src.optimizers.adam import Adam
import numpy as np
np.set_printoptions(precision=2)

data = []
with open('iris.data.txt') as file:
    for line in file.readlines():
        data.append(line.strip().split(','))

flower_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica':2
} 
for i in data:
    for k,j in enumerate(i[:-1]):
        i[k] = float(j)
    i[-1] = flower_map[i[-1]]

norm_data = np.array(data).T
norm_data[:4] = (norm_data[:4]/np.linalg.norm(norm_data[:4]))
norm_data = norm_data.T
np.random.shuffle(norm_data)

topology: list[int] = [4,10,10,3]
learning_rate: float =1e-2
activation_functions = [no_op(),*[ReLu() for i in range(len(topology)-2)],softMax()]
cost_function = CrossEntropy()
optimizer = Adam(topology,learning_rate,0.9,0.999,1e-8)
batch_size = 40
test_size = 20
epochs = 300
net = NeuralNetwork(topology,activation_functions,cost_function,optimizer,learning_rate)
test_data = norm_data[120:]
train_data = norm_data[:120]
 
errors = []
accuraccy = []
for i in range(epochs):
    batch_inputs = np.zeros((batch_size,topology[0]))
    batch_solutions = np.zeros((batch_size,topology[-1]))
    for j in range(batch_size):
        index = random.randint(0,len(train_data)-1)
        batch_inputs[j] = np.array([train_data[index][:4]])
        solutions = np.zeros(3)
        solutions[int(train_data[index][4])] = 1
        batch_solutions[j] = np.array(solutions)
    net.backward_propagation(batch_inputs,batch_solutions)

    test_inputs = np.zeros((test_size,topology[0]))
    test_solutions = np.zeros((test_size,topology[-1]))

    for j in range(test_size):
        index = random.randint(0,len(test_data)-1)
        test_inputs[j] = np.array([test_data[index][:4]])
        solutions = np.zeros(3)
        solutions[int(test_data[index][4])] = 1
        test_solutions[j] = np.array(solutions)
    
    predicted_results = net.feed_forward(test_inputs)
    # print(f'Predicted:{list(flower_map.keys())[np.argmax(predicted_results[3])] }, Actual:{list(flower_map.keys())[np.argmax(test_solutions[3])]}')
    errors.append(cost_function(test_solutions,predicted_results).mean())
    acc = np.where(np.argmax(predicted_results,axis=1)==np.argmax(test_solutions,axis=1))[0].shape[0]/test_size
    accuraccy.append(acc)
    print(f'Epoch:{i},accuracy{round(acc*100,2)}% Error:{round(cost_function(test_solutions,predicted_results).mean(),2)},Predicted:{predicted_results[5]},actual:{test_solutions[5]}')


pyplot.plot(range(len(errors[0:])),errors[0:],label='Errors')
pyplot.plot(range(len(errors[0:])),accuraccy[0:],label='Accuracy')
pyplot.legend()
pyplot.xlabel('Epochs')
pyplot.show()