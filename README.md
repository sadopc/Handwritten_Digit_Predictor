# Handwritten Digit Predictor

Welcome to the Handwritten Digit Predictor! This project allows you to draw a handwritten digit on a canvas and see how a trained Convolutional Neural Network (CNN) model predicts it. It's a fun and interactive way to explore the power of deep learning in recognizing handwritten digits.

## Prerequisites

To use this project, you need the following software and libraries installed:

- Python 3.x
- tkinter (usually comes with Python)
- PIL (Pillow)
- PyTorch
- torchvision

## Installation

Install the required libraries using pip:

`
pip install torch torchvision pillow
`

## Usage

1. **Train the Model**

   Run the following command to train the CNN on the MNIST dataset:

   `
   python model_creation.py
   `   

   This will save the model weights to `mnist_cnn_improved.pth`.

2. **Run the Predictor**

   Launch the GUI by running:

   `
   python Handwritten_Digit_Predictor.py
   `

   A window will appear where you can draw a digit and see the prediction.

## Model Architecture

The CNN model consists of two convolutional layers with batch normalization and ReLU activation, followed by max pooling. This is followed by two fully connected layers with dropout regularization and batch normalization, and a final output layer with 10 units for the digit predictions.

## Training the Model

The script `model_creation.py` trains the model for 15 epochs using an Adam optimizer and step learning rate scheduler. Data augmentation techniques like random affine transformations are used to improve generalization.

## Running the Predictor

In the predictor, the drawn image is inverted, resized to 28x28 pixels, and normalized before being fed into the model for prediction. The GUI displays the probabilities for each digit in progress bars.

## Acknowledgments

Thanks to the MNIST dataset for providing the handwritten digits. Built with PyTorch and Tkinter.

## Notes

- The model will train faster if you have a GPU. The script will automatically use it if available.
- The predictor will work on CPU if GPU is not available.

We hope you find this project insightful and fun to use!
