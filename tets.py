import numpy as np
from time import perf_counter
test_cases = 10
input_neurons = 10000
first_layer_neurons = 5000

a = np.random.rand(test_cases,input_neurons)
b = np.random.rand(first_layer_neurons,input_neurons)

# a1 = perf_counter()
# j = np.dot(b,np.transpose(a))
# a2 = perf_counter()
# print(f'First time={a2-a1}')
a1 = perf_counter()
d = np.dot(a,np.transpose(b))
a2 = perf_counter()
print(f'Second time={a2-a1}')