from . import Layer , Sequential , Module , Softmax

class Model:
    def __init__(self,shape:list,activation:Module,epoch=5) -> None:
        self.layers = Sequential()
        self.epoch = epoch
        n=1
        while(n<len(shape)):
            self.layers.layers.append(Layer(shape[n-1],shape[n]))
            if n!=len(shape)-1:
                self.layers.layers.append(activation.__class__())
        self.criterion = Softmax()
    def fit(self,X,Y):
        for i in range(self.epoch):
            X_permutaion
            Y_permutaion