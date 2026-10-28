from . import Layer , Sequential , Module , SoftmaxCrossEntropy
from optimizer import SGD
import numpy as np

class Modal:
    def __init__(self,shape:list,activation,optimizer=SGD) -> None:
        self.modal = Sequential()
        n=1
        while(n<len(shape)):
            self.modal.layers.append(Layer(shape[n-1],shape[n]))
            if n !=len(shape)-1:
                self.modal.layers.append(activation())
            n+=1
        
        self.criterion = SoftmaxCrossEntropy()
        self.optimizer = optimizer(self)
    def fit(self,X,Y,batch_size = 64):
        n= X.shape[0] # len of input data 
        total_loss = 0.0
        for start in range(0,n, batch_size):
            # self.optimizer.zero_grad()
            end = min(start + batch_size,n)
            X_batch = X[start:end,:]
            Y_batch = Y[start:end,:]

            Y_hat= self.modal.forward(X_batch)
            loss = self.criterion.forward(Y_hat,Y_batch)
            total_loss += loss
            grad = self.criterion.backward()
            self.modal.backward(grad)
            self.optimizer.step()
        return total_loss
    def predict(self,X):
        return self.modal.forward(X)
            