from .base import Module
import numpy as np
# x shape (1,ip) each input is a row vector , this can also be like (batch, ip)
# w shape (ip,op) each nneuron weights is a column vector
# b shape (1,op) 
# z shape (1,op)
# (1,ip) X (ip,op) = output(1,op)
class Layer(Module):
    def __init__(self,ip,op) -> None:
        self.W:np.ndarray = np.random.rand(ip,op)  * np.sqrt(2.0 / ip)
        self.b:np.ndarray = np.zeros((1,op))

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
    def set_grad(self,dW,db) -> None:
        self.dW = dW
        self.db = db
    def forward(self,x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, grad_out:np.ndarray):
        self.dW += self.x.T @ grad_out # (ip,op) += (ip,batch) X (batch,op)
        self.db += grad_out.sum(axis=0) #(batch,op) += sum(op,from 0 to batch)
        # dL\dx = dL\dy * (df/dx + dg\dx) if y = f(x) + g(x)
        grad_in = grad_out @ self.W.T # (batch,ip) = (batch,op) X (op,ip)
        return grad_in
    def parameters(self):
        return [self.W , self.b]
    def grad(self):
        return [self.dW, self.db]
