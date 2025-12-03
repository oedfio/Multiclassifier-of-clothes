# Fashion MNIST Multiclassifier

A deep learning project for multi-class clothing classification using the Fashion MNIST dataset. This project trains a neural network to classify images of clothing items into 10 different categories.

## Project Overview

This project implements a neural network classifier that can recognize and categorize clothing items from 28x28 pixel grayscale images. The model is trained on the Fashion MNIST dataset, which contains 70,000 images across 10 clothing categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Features

- **Neural Network Model**: A 4-layer fully connected neural network with ReLU activations
- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **GPU Support**: Automatically detects and uses CUDA if available
- **Fashion MNIST Dataset**: Automatic dataset downloading and preprocessing
- **Model Persistence**: Saves trained model for future inference

## Requirements

- Python 3.7+
- PyTorch
- TorchVision
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Multiclassifier-of-clothes
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── main.py              # Main entry point for training and testing
├── model.py             # Neural network model definition
├── train.py             # Training logic and data loading
├── test_model.py        # Model evaluation and testing
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## Usage

### Training the Model

Run the main script to train the model and save it:

```bash
python main.py
```

This will:
1. Check for GPU availability
2. Train the model for 100 epochs
3. Save the trained model as `fashion_mnist_model.pth`
4. Evaluate the model on test data

### Custom Training

To modify training parameters, edit `main.py` and adjust:
- `epochs`: Number of training epochs (default: 100)
- `device`: Force CPU or CUDA usage

## Model Architecture

The `FashionClassifier` network consists of:

```
Input (784 features - 28x28 flattened image)
    ↓
Linear(784 → 256) + ReLU
    ↓
Linear(256 → 128) + ReLU
    ↓
Linear(128 → 64) + ReLU
    ↓
Linear(64 → 10 output classes)
```

## Training Details

- **Dataset Split**: 50,000 training samples, 10,000 validation samples
- **Batch Size**: 64
- **Early Stopping**: Patience of 10 epochs
- **Normalization**: Mean=0.5, Std=0.5
- **Optimizer**: Available in `train.py`
- **Loss Function**: Available in `train.py`

## Output

After training completes:
- The trained model is saved as `fashion_mnist_model.pth`
- Test metrics are printed to console
- Model performance statistics are displayed

## Performance

The model achieves high accuracy on the Fashion MNIST dataset through:
- Multi-layer architecture for feature extraction
- ReLU activations for non-linearity
- Early stopping to prevent overfitting


## Author

Created by: oedfio