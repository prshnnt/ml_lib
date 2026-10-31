from neural_network import Sigmoid
from neural_network.criterion import MSELoss
import numpy as np
import pandas as pd
from neural_network import Modal
from neural_network.activation import ReLU , LeakyReLU ,Sigmoid
from optimizer import SGD

train = pd.read_csv("./sample_data/mnist_train_small.csv", header=None)
test = pd.read_csv("./sample_data/mnist_test.csv", header=None)

lr = 0.01
beta = .9
EPOCHS = 20
batch_size = 32


def train_modal(modal:Modal):
    X = train.iloc[:,1:].to_numpy(dtype=np.float32) / 255.0
    Y = train.iloc[:,1:].to_numpy(dtype=np.float32) / 255.0
    for epoch in range(EPOCHS):
        permutation = np.random.permutation(X.shape[0])
        X_all = X[permutation,:]
        Y_all = Y[permutation]
        loss = modal.fit(X_all,Y_all)
        print(f"epoch: {epoch}  loss:{loss:.5f}")
def test_modal(modal:Modal):
    X = test.iloc[:,1:].to_numpy(dtype=np.float32) /255.0
    Y = test.iloc[:,1:].to_numpy(dtype=np.float32) /255.0

    logits = modal.predict(X)
    loss = modal.criterion.forward(logits , Y)
    print(f"Test Loss : {loss:.4f}")
import matplotlib.pyplot as plt

def evaluate_reconstructions(modal: Modal, test_data, num_samples=5):
    X = test_data.iloc[:num_samples, 1:].to_numpy(dtype=np.float32) / 255.0
    reconstructed = modal.predict(X)

    fig, axes = plt.subplots(2, num_samples, figsize=(10, 4))
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(X[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Original")
        # Reconstructed
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Reconstructed")
    plt.show()

if __name__ == "__main__":
    modal = Modal(shape=[784,128,50,128,784],activation=LeakyReLU,criterion=MSELoss,optimizer=SGD)
    modal.modal.layers.append(LeakyReLU())
    train_modal(modal)
    # test_modal(modal)
    evaluate_reconstructions(modal,test)
    modal.save("autoencoder.pkl")
    