import numpy as np

class Module:
    def forward(self,x):
        raise NotImplementedError
    def backward(self,grad_out):
        raise NotImplementedError
    def parameters(self):
        return []
    def grad(self):
        return []

class Sequential(Module):
    def __init__(self,*layers):
        self.layers:list[Module] = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x= layer.forward(x)
        return x
    def backward(self, grad_out):
        for layer in reversed(self.layers):
            grad_out = layer.backward(grad_out)
        return grad_out
    def parameters(self):
        return [
            p for layer in self.layers 
            for p in layer.parameters()
        ]
    def grad(self):
        return [
            p for layer in self.layers 
            for p in layer.grad()
        ]
    