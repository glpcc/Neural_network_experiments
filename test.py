import numpy as np
from timeit import timeit

a = np.array([[0,1],[0,1],[0,1],[1,0]])
b = np.array([[0.2,0.5],[0,1],[1,0],[1,0]])
print(np.where(np.argmax(a,axis=1)==np.argmax(b,axis=1))[0].shape[0]/len(a))