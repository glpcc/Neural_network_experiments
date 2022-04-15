from src.neural_network import NeuralNetwork


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

print(data)

