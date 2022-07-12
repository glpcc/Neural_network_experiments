# INTRODUCTION
## Motivation
This is a personal project to understand more deeply how neural networks work and maybe later test with edge cases like self modifying networks.

### Achieved goals
I've achieved to make a model to predict the types of flowers in the [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) as you can see in the following [graph](https://github.com/glpcc/Neural_network_experiments/blob/master/img/Iris%20Data%20prediction.png).\

I've also achieved to correctly identify the numbers in the images of the [Mnist Dataset](https://en.wikipedia.org/wiki/MNIST_database) with 28x28 images that look like [this](img/mnist_images.png). Each model stored as a pickle obj in the models folder achieved the accuracy in the final test that is shown in the name. [Here](img/Mnist_model_accuraccy_learning2.png) is the graph of the learning of one of them. Theres is also a interactive teste of model in which you draw numbers in a pygame window and when you press enter it tries to guess the number drawn (It isnt very good for certain numbers) the test is in the file *mnist_model_tester.py* follow the installation instrucctions to use it


## Installation and usage 
### To use the Neural network class
```shell
pip install -r requirements.txt
```
### To use the mnist model teste
```shell
pip install -r requirements_mnist_model_tester.txt
```
### To train a model for the mnist dataset
```shell
pip install -r requirements_mnist_model_train.txt
```

Then run the python file that you want to test