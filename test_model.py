import matplotlib.pyplot as plt
import torch
import os
import numpy as np

from torchvision import datasets, transforms
from model import FashionClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fashion_mnist_model.pth")
DATASET_DIR = os.path.dirname(os.path.abspath(__file__))

TEXT_LABELS = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

def test_model(device="cpu"):
    print("Testing...")
    model = FashionClassifier()

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_set = datasets.FashionMNIST(
        root=DATASET_DIR,
        train=True,
        transform=transform,
        download=True
    )
    test_loader=torch.utils.data.DataLoader(
        test_set,    
        batch_size=64,   
        shuffle=True)

    plt.figure(dpi=300, figsize=(5,1))
    for i in range(5):
        ax = plt.subplot(1, 5, i + 1)
        img = test_set[i][0]
        img = img / 2 + 0.5
        img = img.reshape(28, 28)
        plt.imshow(img, cmap="binary")
        plt.axis("off")
        img, label = test_set[i]
        img = img.reshape(-1, 28*28).to(device)
        pred = model(img)
        index_pred = torch.argmax(pred, dim = 1)
        idx = index_pred.item()
        plt.title(f"l/p {TEXT_LABELS[label]}/{TEXT_LABELS[idx]}", fontsize=5)

    plot_path = os.path.join(BASE_DIR, "plot.png")
    plt.savefig(plot_path)

    results = []

    for imgs, labels in test_loader:
        imgs = imgs.reshape(-1, 28*28).to(device)
        labels = (labels).reshape(-1,).to(device)
        preds = model(imgs)
        pred10 = torch.argmax(preds, dim = 1)
        correct = (pred10 == labels)
        results.append(correct.detach().cpu().numpy().mean())

    accuracy = np.array(results).mean()
    print(f"The accuracy of the predictions is {accuracy}")