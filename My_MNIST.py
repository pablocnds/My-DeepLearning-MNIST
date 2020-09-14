from pathlib import Path
import requests
import pickle
import gzip
import numpy as np

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F

import cnn



# Function to calculate the loss of a model and apply backpropagation if wanted
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    return loss.item(), len(xb)

# BASIC FIT FUNCTION
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(dev)
            yb = yb.to(dev)
            loss_batch(model, loss_func, xb, yb, opt)
        
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb.to(dev), yb.to(dev)) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)



# Download the MNIST number dataset
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

# Check if its already downloaded
if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)


# Load dataset
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")


# Transform data to processable torch tensors
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()


# Wrap in torch tensor dataset
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)


# GPU
# pylint: disable=E1101
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print(dev)
# pylint: enable=E1101


# MODEL FITTING
bs = 64
lr = 0.1
epochs = 3

loss_func = F.cross_entropy

train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

model = cnn.CNN()
model.to(dev)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)


fit(epochs, model, loss_func, opt, train_dl, valid_dl)


# SHOW RESULTS
import matplotlib.pyplot as plt
import math

# pylint: disable=E1101
model.to(torch.device("cpu"))
# pylint: enable=E1101

for xb, yb in valid_dl:
    for i in range(10):
        plt.imshow(xb[i].reshape((28,28)), cmap="gray")
        guess = model.forward(xb)[i]
        plt.gcf().text(0.02, 0.55, "guess: " + str(max(range(len(guess)), key=guess.__getitem__)), fontsize=14)
        plt.gcf().text(0.02, 0.45, "number: " + str(yb[i].item()), fontsize=14)
        plt.show()