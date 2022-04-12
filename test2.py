import numpy as np
from src.activation_functions.activation_functions import *
a = np.array([[1,2,3],[2,4,5],[3,56,7]])
b = np.array([[1,1,1],[2,2,2],[3,3,3]])

print((np.square(a - b)).mean(axis=1))

