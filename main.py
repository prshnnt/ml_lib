import numpy as np
import pandas as pd
from neural_network import Modal
from neural_network.activation import ReLU

train = pd.read_csv("./sample_data/mnist_train_small.csv", header=None)
test = pd.read_csv("./sample_data/mnist_test.csv", header=None)

lr = 0.01
beta = .9
EPOCHS = 5
batch_size = 32


def train_modal(modal:Modal):
    X = train.iloc[:,1:].to_numpy(dtype=np.float16) / 255.0
    Y = train.iloc[:,0].to_numpy(dtype=np.int16).reshape(len(train),1)
    for epoch in range(EPOCHS):
        permutation = np.random.permutation(X.shape[0])
        X_all = X[permutation,:]
        Y_all = Y[permutation]
        loss = modal.fit(X_all,Y_all)
        print(f"epoch: {epoch}  loss:{loss:.5f}")
def test_modal(modal:Modal):
    X = test.iloc[0,1:].to_numpy(dtype=np.float16) /255.0
    result = np.argmax(modal.predict(X))
    print("Prediction",result)
    print("Original",test.iloc[0,0])


if __name__ == "__main__":
    modal = Modal(shape=[784,128,10],activation=ReLU)
    train_modal(modal)
    test_modal(modal)
    