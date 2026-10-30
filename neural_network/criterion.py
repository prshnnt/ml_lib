import numpy as np
class Criterion:
    def forward(self,logits,targets):
        pass
    def backward(self):
        pass
class SoftmaxCrossEntropy(Criterion):
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

class MSELoss(Criterion):
    def forward(self, y_pred, y_true):
        self.diff = y_pred - y_true
        self.batch_size = y_pred.shape[0]
        self.features = y_pred.shape[1]
        return np.mean(self.diff ** 2)

    def backward(self):
        # Gradient of mean squared error wrt predictions
        return 2.0 * self.diff / (self.batch_size * self.features)
