from abc import ABC, abstractmethod
import numpy as np

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self,array: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def calculate_errors(self,prev_errors: np.ndarray,weighted_inputs: np.ndarray) -> np.ndarray:
        ... 


class no_op(ActivationFunction):
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return array
    
    def calculate_errors(self,prev_errors: np.ndarray,weighted_inputs: np.ndarray) -> np.ndarray:
        return prev_errors


class tanh(ActivationFunction):
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return np.tanh(array)
    
    def calculate_errors(self,prev_errors: np.ndarray,weighted_inputs: np.ndarray) -> np.ndarray:
        b = np.tanh(weighted_inputs)
        return prev_errors*(1 - b*b)

class sigmoid(ActivationFunction):
    def __call__(self,array: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-array))
    
    def calculate_errors(self,prev_errors: np.ndarray,weighted_inputs: np.ndarray) -> np.ndarray:
        return prev_errors*(self(weighted_inputs)*(1-self(weighted_inputs)))


class softMax(ActivationFunction):
    def __call__(self, array: np.ndarray) -> np.ndarray:
        maxs = array.max(axis=1,keepdims=True)
        exps = np.exp(array - maxs)
        sums = np.sum(exps,axis=1,keepdims=True)
        return exps / sums

    def calculate_errors(self,prev_errors: np.ndarray,weighted_inputs: np.ndarray) -> np.ndarray:
        errors = np.zeros(prev_errors.shape)
        softmax = self(weighted_inputs)
        for i in range(prev_errors.shape[0]):
            temp = np.reshape(softmax[i], (1, -1))
            errors[i] = prev_errors[i].reshape((1, -1))@(temp * np.identity(temp.size) - temp.transpose() @ temp)
        return errors

class ReLu(ActivationFunction):
    def __call__(self, array: np.ndarray) -> np.ndarray:
        return array*(array>0)

    def calculate_errors(self,prev_errors: np.ndarray,weighted_inputs: np.ndarray) -> np.ndarray:
        return prev_errors*(1*(weighted_inputs>0))


s = softMax()
n2 = np.array([[0,0,0],[1,4,3]])
n3 = np.array([[1,1,1],[1,1,1]])
print(s(n2))
print(s.calculate_errors(n2,n3))
