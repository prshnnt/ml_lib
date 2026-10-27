from .base import Module
import numpy as np

class Layer(Module):
    def __init__(self,ip,op) -> None:
        self.W:np.ndarray = np.random.rand(op,ip)  * np.sqrt(2.0 / ip)
        self.b:np.ndarray = np.zeros((op,1))

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
    def set_grad(self,dW,db) -> None:
        self.dW = dW
        self.db = db
    def forward(self,x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, grad_out:np.ndarray):
        self.dW += self.x.T @ grad_out
        self.db += grad_out.sum(axis=0)
        grad_in = grad_out @ self.W.T
        return grad_in
    def parameters(self):
        return [self.dW , self.db]
