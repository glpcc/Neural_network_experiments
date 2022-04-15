import numpy as np

n1 = np.array([[1,2],[3,2],[1,3],[1,1]])
n2 = np.array([[[1,2],[3,2]],[[1,1],[1,1]],[[1,2],[3,2]],[[2,2],[2,2]]])
print(np.tensordot(n2,n1,axes=0))
# print(np.einsum('ji,mik->jmk',n1,n2))


n1 = np.array([3,2])
n2 = np.array([[1,1],[1,1]])
print(np.einsum('i,ij->j',n1,n2))
print(np.dot(n1,n2))