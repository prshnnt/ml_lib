class SGD:
    def __init__(self,modal,lr=0.01):
        self.modal = modal
        self.parameters = self.modal.modal.parameters()
        self.lr = lr
    def step(self):
        # Compute gradients for all parameters
        grads = self.modal.modal.grad()
        # Update each parameter in-place and reset its gradient
        for param, grad in zip(self.parameters, grads):
            param -= self.lr * grad
            # Zero out the gradient after applying the update
            grad[:] = 0