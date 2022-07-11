from cProfile import label
from keras.datasets import mnist
from src.activation_functions.softmax import SoftMax
from src.cost_funtions.cost_function import CostFunction
from src.layers.layer import Layer
from src.neural_network import NeuralNetwork
from src.layers.convolutional2D import Convolutional2D
from src.layers.dense import Dense
from src.optimizers.adam import Adam
from src.activation_functions.relu import ReLu
from src.activation_functions.no_op import No_op
from src.cost_funtions.categorical_cross_entropy import CategoricalCrossEntropy
import numpy as np
from matplotlib import pyplot


(train_x, train_y), (test_x, test_y) = mnist.load_data()

first_filter_shape = (3,3)
second_filter_shape = (3,3)
num_filters = 1
image_shape = (28,28)
second_image_shape = (image_shape[0]-first_filter_shape[0]+1,image_shape[1]-first_filter_shape[1]+1) #(26,26)
first_dense_inputs = (second_image_shape[0]-second_filter_shape[0]+1)*(second_image_shape[1]-second_filter_shape[1]+1) # 24*24 = 576
cost_function: CostFunction = CategoricalCrossEntropy()
layers: list[Layer] = [
    Convolutional2D(ReLu,Adam,num_filters,first_filter_shape,image_shape,learning_rate=1e-4),
    Dense(ReLu,Adam, 26*26*num_filters, 100, learning_rate=1e-4),
    Dense(SoftMax,Adam,100,10,learning_rate=1e-4 )
]
net = NeuralNetwork(layers,cost_function)

batch_size = 10
number_of_batches_per_epoch = 50
number_of_epochs = 100
train_index = 0
test_index = 0
test_size = 20
all_errors = []
all_accuracy = []
for epoch_number in range(number_of_epochs):
    for batch_number in range(number_of_batches_per_epoch):
        # TODO Maybe make it random 
        batch_x = train_x[train_index:train_index + batch_size]
        batch_solutions = train_y[train_index:train_index + batch_size]
        batch_y = np.zeros((len(batch_solutions),10))
        for i,j in enumerate(batch_y):
            j[batch_solutions[i]] = 1
        net.backward_propagation(batch_x,batch_y)
        train_index += batch_size

    net_test_x = test_x[test_index: test_index+test_size]
    net_test_solutions = test_y[test_index: test_index+ test_size]
    net_test_y = np.zeros((len(net_test_solutions),10))
    for i,j in enumerate(net_test_y):
        j[net_test_solutions[i]] = 1
    errors, output = net.test(net_test_x,net_test_y)
    print(f'Epoch:{epoch_number} Output:{output[0]} Actual:{net_test_y[0]} \n Error:{errors.mean()}')
    all_errors.append(errors.mean())
    acc = 0
    for i, out in enumerate(output): 
        if np.argmax(out) == np.argmax(net_test_y[i]):
            acc += 1
    all_accuracy.append(acc/test_size)
    test_index += test_size 

# Big Test of final verison
b_test_size = 100
acc = 0
for i in range(b_test_size):
    test_ind: int = np.random.choice(test_x.shape[0])
    t_x = test_x[test_ind].reshape((1,*test_x[test_ind].shape))
    t_y = test_y[test_ind]
    output = net.feed_forward(t_x)
    if np.argmax(output) == t_y:
        acc += 1

print(f'Big test acc: {acc/b_test_size}')


print(f'Average acc:{np.mean(all_accuracy)}, Average err:{np.mean(all_errors)}')
pyplot.plot(list(range(number_of_epochs)), all_errors, label='errors')
pyplot.plot(list(range(number_of_epochs)), all_accuracy, label='accuracy')
pyplot.legend()
pyplot.show()