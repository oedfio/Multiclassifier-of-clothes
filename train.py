import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FashionClassifier

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = BASE_DIR

class EarlyStop:
    def __init__(self, patience=10):
        self.patience = patience
        self.steps = 0
        self.min_loss = float('inf')
    def stop(self, val_loss):
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.steps = 0
        elif val_loss >= self.min_loss:
            self.steps += 1
        if self.steps >= self.patience:
            return True
        else:
            return False

def train_model(device="cpu", epochs=5):

    #----Dataset----
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_set = datasets.FashionMNIST(
        root=DATASET_DIR,
        train=True,
        transform=transform,
        download=True
    )

    train_set,val_set=torch.utils.data.random_split(train_set,[50000,10000])
    train_loader=torch.utils.data.DataLoader(
        train_set,    
        batch_size=64,   
        shuffle=True)   
    val_loader=torch.utils.data.DataLoader(
        val_set,    
        batch_size=64,   
        shuffle=True)

    #----Model----
    model = FashionClassifier().to(device)

    #----Loss + Optimizer + Stopper----
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    stopper = EarlyStop()

    #----Training----
    print("Training...")
    for epoch in range(epochs):
        total_loss = 0
        
        for n, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.reshape(-1,28*28).to(device)
            labels = labels.reshape(-1,).to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()

        total_loss /= n 

        val_loss = 0
        for n, (imgs, labels) in enumerate(val_loader):
            imgs = imgs.reshape(-1,28*28).to(device)
            labels = labels.reshape(-1,).to(device)
            preds = model(imgs)
            loss = loss_fn(preds, labels)
            val_loss += loss.detach()

        val_loss /= n

        print(f"at epoch {epoch}, total_loss is {total_loss}, val_loss is {val_loss}")

        if stopper.stop(val_loss)==True:
            break

    return model