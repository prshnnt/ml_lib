from .base import Module
import numpy as np
# this is teh formula for derivative and chain rule
# y = a(x)
# dy\dx = da\dx
# dL\dx = (dL\dy)*(dy\dx)

class ReLU(Module):
    def forward(self,x):
        self.mask = x > 0
        return x * self.mask
    def backward(self, grad_out):
        return grad_out * self.mask # grad input

class Sigmoid(Module):
    def forward(self, x):
        self.y = 1 / (1+np.exp(-x))
        return self.y
    def backward(self, grad_out):
        derivative = self.y * ( 1 - self.y)
        return grad_out * derivative # grad input

class Softmax(Module):
    def forward(self, x):
        x_hat = x - np.max(x, axis=-1 , keepdims=True) # shifted x
        exp_x = np.exp(x_hat)
        self.y = exp_x / np.sum(
            exp_x , axis= -1 , keepdims=True
        )
        return self.y
    def backward(self, grad_out):
        #grad_out= gi = dL\dyi
        # dL\dxj = yi ( gj - sum(gi, yi))
        dot = np.sum(
            grad_out * self.y,
            axis=-1,
            keepdims=True
        )
        return self.y * (grad_out - dot) # grad input

class LeakyReLU(Module):
    alpha=0.01
    def forward(self, x):
        self.mask = (x > 0) +( self.alpha*(x<0))
        return x * self.mask
    def backward(self, grad_out):
        return grad_out * self.mask