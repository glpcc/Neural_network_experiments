import numpy as np
from time import perf_counter

# avrg1 = 0
# avrg2 = 0
# for i in range(10):
#     print(i)
#     n1 = np.random.randn(5000,5000) - 0.5
#     a1 = perf_counter()
#     n1 = n1 * (n1>0)
#     a2 = perf_counter()
#     avrg1 += a2-a1
#     print(a2-a1)
#     n2 = np.random.randn(5000,5000) - 0.5
#     a1 = perf_counter()
#     np.maximum(n2,0,n2)
#     a2 = perf_counter()
#     avrg2 += a2-a1
#     print(a2-a1)
# print(avrg1/100)
# print(avrg2/100)

n1 = np.array([[-2,4],[3,-8]])
n2 = np.array([[-2,4],[3,-8]])
print(n2*(1*(n1>0)))