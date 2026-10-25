import numpy as np
from abc import ABC

class Activation(ABC):
    def forward(self,X):
        pass
    def derivative(self,X):
        pass

class ReLU(Activation):
    def forward(self, Z:np.ndarray):
        return np.maximum(0, Z)
    def derivative(self, Z:np.ndarray):
        return Z>0
    
class Layer:
    def __init__(self,ip,op) -> None:
        self.W = np.random.rand(op,ip)  * np.sqrt(2.0 / ip)
        self.b = np.zeros((op,1))
    def forward(self,X):
        self.X = X
        self.Z = (self.W @ self.X.T) + self.b
        return self.Z

class Sequence:
    def __init__(self, sequence = []) -> None:
        self.layers = sequence
    def forword(self,X):
        self.output = X
        for layer in self.layers:
            self.output = layer.forward(self.output)
        return self.output
    def backward(self,delta):
        for layer_index in reversed(range(len(self.layers))):
            layer = self.layers[layer_index]
            dW = delta @ layer.X.T 
            db = delta
            
