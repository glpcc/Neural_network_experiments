from keras.datasets import mnist
from src.activation_functions.softmax import SoftMax
from src.cost_funtions.cost_function import CostFunction
from src.layers.layer import Layer
from src.neural_network import NeuralNetwork
from src.layers.convolutional2D import Convolutional2D
from src.layers.dense import Dense
from src.optimizers.adam import Adam
from src.activation_functions.relu import ReLu
from src.cost_funtions.categorical_cross_entropy import CategoricalCrossEntropy
import numpy as np
from matplotlib import pyplot
import pickle

(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x/255
test_x = test_x/255
first_filter_shape = (4,4)
num_filters = 6
image_shape = (28,28)
cost_function: CostFunction = CategoricalCrossEntropy()
layers: list[Layer] = [
    Convolutional2D(ReLu,Adam,num_filters,first_filter_shape,image_shape,learning_rate=1e-5),
    Dense(ReLu,Adam, 25*25*num_filters, 500, learning_rate=1e-4),
    Dense(SoftMax,Adam,500,10,learning_rate=1e-4)
]
net = NeuralNetwork(layers,cost_function)

batch_size = 20
number_of_batches_per_epoch = 10
number_of_epochs = 800
train_index = 0
test_index = 0
test_size = 20
all_errors = []
all_accuracy = []
for epoch_number in range(number_of_epochs):
    for batch_number in range(number_of_batches_per_epoch):
        batch_ind = np.random.choice(train_x.shape[0],size=batch_size)
        batch_x = np.take(train_x,batch_ind, axis=0)
        batch_solutions = np.take(train_y, batch_ind, axis=0)
        batch_y = np.zeros((len(batch_solutions),10))
        for i,j in enumerate(batch_y):
            j[batch_solutions[i]] = 1
        net.backward_propagation(batch_x,batch_y)
        train_index += batch_size

    net_test_ind = np.random.choice(test_x.shape[0],size=batch_size)
    net_test_x = np.take(test_x,net_test_ind, axis=0)
    net_test_solutions = np.take(test_y, net_test_ind, axis=0)
    net_test_y = np.zeros((len(net_test_solutions),10))
    for i,j in enumerate(net_test_y):
        j[net_test_solutions[i]] = 1
    errors, output = net.test(net_test_x,net_test_y)
    all_errors.append(errors.mean())
    acc = 0
    for i, out in enumerate(output): 
        if np.argmax(out) == np.argmax(net_test_y[i]):
            acc += 1
    all_accuracy.append(acc/test_size)
    print(f'Epoch:{epoch_number} Acc:{acc/test_size} Err:{errors.mean()}')
    test_index += test_size 

# Big Test of final verison
b_test_size = 1000
acc = 0
for i in range(b_test_size):
    test_ind: int = np.random.choice(test_x.shape[0])
    t_x = test_x[test_ind].reshape((1,*test_x[test_ind].shape))
    t_y = test_y[test_ind]
    output = net.feed_forward(t_x)
    if np.argmax(output) == t_y:
        acc += 1

print(f'Big test acc: {acc/b_test_size}')
if acc/b_test_size > 0.6 :
    f = open(f'models/mnist_predictor_acc{acc/b_test_size}.obj', 'wb')
    pickle.dump(net,f)

print(f'Average acc:{np.mean(all_accuracy)}, Average err:{np.mean(all_errors)}')
# pyplot.plot(list(range(number_of_epochs)), all_errors, label='errors')
pyplot.plot(list(range(number_of_epochs)), all_accuracy, label='accuracy')
pyplot.legend()
pyplot.show()