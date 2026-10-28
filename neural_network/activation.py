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
        derivative = self.y ( 1 - self.y)
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

class SoftmaxCrossEntropy:
    def forward(self,logits,targets):
        # logits shape (batch , no of classes/op)
        # targets shape (batch,) containing class indices
        self.targets =targets
        self.batch_size = logits.shape[0]

        shifted_logits = logits - np.max(logits,axis=1 , keepdims=True)
        exp_logits = np.exp(shifted_logits)

        self.prob = exp_logits / np.sum(exp_logits,axis=1,keepdims=True)

        # CE loss using log-sum-exp
        log_sum_exp = np.log(
            np.sum(exp_logits,axis=1)
        )+np.max(logits,axis=1)

        correct_logits = logits[
            np.arange(self.batch_size), targets
        ]
        loss = np.mean( log_sum_exp - correct_logits)
        return loss
    def backward(self):
        # returns gradiant of mean loss wrt logits
        # this is like y_hat - y where targets are indexes of values , instead of doing one hot encoding and then subtracting from prediction we just subtract that value
        grad = self.prob.copy()
        grad[np.arange(self.batch_size),self.targets] -=1
        grad/=self.batch_size
        return grad