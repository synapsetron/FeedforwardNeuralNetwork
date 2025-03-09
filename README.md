# Neural Network Trainer

## Overview
This project implements a simple neural network trainer to approximate mathematical functions using PyTorch. It allows training different network architectures and activation functions to analyze their impact on performance.

## Features
- Customizable neural network architecture with varying hidden layers and neurons per layer.
- Support for multiple activation functions (`ReLU`, `Sigmoid`, `Tanh`, `LeakyReLU`).
- Visualization of:
  - Training loss over epochs for different architectures.
  - Training loss for different activation functions.
  - Function approximation performance with different architectures.
  - Function approximation with different activation functions.

## Installation

### Prerequisites
Ensure you have Python installed along with the required dependencies:

```bash
pip install torch numpy matplotlib
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/neural-network-trainer.git
cd neural-network-trainer
```

2. Run the script to train and visualize results:

```bash
python neural_network_trainer.py
```

## Code Structure
- `NeuralNetwork`: Defines a simple feedforward neural network with customizable architecture.
- `NeuralNetworkTrainer`: Handles training, loss tracking, and visualization.
- Main script (`if __name__ == "__main__"`): Defines the data and executes training and visualization functions.

## Examples

### Training Loss Comparison for Different Architectures
The script plots the loss curve over training epochs for various neural network architectures to compare performance.

### Activation Function Impact
The script visualizes how different activation functions affect training loss and function approximation.

### Function Approximation
Visualizes the ability of the trained neural networks to approximate mathematical functions such as:
- \( f(x) = x^2 \)
- \( f(x) = x^3 + 2x \)

