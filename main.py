import torch 
import os

from train import train_model
from test_model import test_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = train_model(device=device, epochs=100)

    save_path = os.path.join(BASE_DIR, "fashion_mnist_model.pth")
    torch.save(model.state_dict(), save_path)
    print("Model saved as fashion_mnist_model.pth")
    test_model(device)

if __name__=="__main__":
    main()