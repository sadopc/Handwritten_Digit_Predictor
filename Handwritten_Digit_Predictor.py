#Handwritten Digit Predictor.py
import os
import tkinter as tk  # For creating the GUI
from tkinter import ttk  # For themed widgets in the GUI
from PIL import Image, ImageDraw  # For image processing
import torch  # For deep learning models
import torch.nn as nn  # For neural network layers
import numpy as np  # For numerical operations

# Define the CNN model with batch normalization and dropout
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer: 1 input channel, 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Batch normalization for the first convolutional layer
        self.bn1 = nn.BatchNorm2d(6)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Batch normalization for the second convolutional layer
        self.bn2 = nn.BatchNorm2d(16)
        # First fully connected layer: 16*4*4 inputs, 120 outputs
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # Batch normalization for the first fully connected layer
        self.bn3 = nn.BatchNorm1d(120)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        # Second fully connected layer: 120 inputs, 84 outputs
        self.fc2 = nn.Linear(120, 84)
        # Batch normalization for the second fully connected layer
        self.bn4 = nn.BatchNorm1d(84)
        # Output layer: 84 inputs, 10 outputs (for digits 0-9)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Apply first convolutional layer, batch normalization, ReLU activation, and pooling
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        # Apply second convolutional layer, batch normalization, ReLU activation, and pooling
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 16 * 4 * 4)
        # Apply first fully connected layer, batch normalization, and ReLU activation
        x = nn.functional.relu(self.bn3(self.fc1(x)))
        # Apply dropout to regularize the model
        x = self.dropout(x)
        # Apply second fully connected layer, batch normalization, and ReLU activation
        x = nn.functional.relu(self.bn4(self.fc2(x)))
        # Apply the output layer
        x = self.fc3(x)
        return x

# Load the trained model
net = Net()
net.load_state_dict(torch.load('mnist_cnn_improved.pth', map_location=torch.device('cpu')))
net.eval()  # Set the model to evaluation mode

# Create the main window
root = tk.Tk()
root.title("Handwritten Digit Predictor")

# Canvas for drawing
canvas_size = 280
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg='white')
canvas.grid(row=0, column=0, rowspan=2)

# Variables to store drawing data
img = Image.new('L', (canvas_size, canvas_size), 255)  # Create a white image
draw = ImageDraw.Draw(img)

# Flags
prediction_loop_running = False  # To control the prediction loop

# Function to start the prediction loop
def start_prediction_loop():
    if prediction_loop_running:
        predict_number()
        root.after(100, start_prediction_loop)  # Run predict_number every 100ms

# Function to start drawing
def start_drawing(event):
    global prediction_loop_running
    if not prediction_loop_running:
        prediction_loop_running = True
        start_prediction_loop()

# Function to stop drawing
def stop_drawing(event):
    global prediction_loop_running
    prediction_loop_running = False

# Function to draw on canvas
def draw_on_canvas(event):
    x = event.x
    y = event.y
    canvas.create_oval(x-10, y-10, x+10, y+10, fill='black', outline='black')
    draw.ellipse((x-10, y-10, x+10, y+10), fill='black', outline='black')

# Bind events to the canvas
canvas.bind('<Button-1>', start_drawing)  # Start drawing when left mouse button is pressed
canvas.bind('<ButtonRelease-1>', stop_drawing)  # Stop drawing when left mouse button is released
canvas.bind('<B1-Motion>', draw_on_canvas)  # Draw when left mouse button is moved

# Function to predict the number with improved preprocessing
def predict_number():
    # Capture the canvas as a NumPy array
    img_array = np.array(img)
    # Invert the image (black on white to white on black)
    img_inverted = 255 - img_array
    # Resize the image to 28x28 pixels, which is the input size for the model
    img_resized = Image.fromarray(img_inverted).resize((28, 28), resample=Image.BICUBIC)
    # Convert the resized image to a NumPy array
    img_resized_array = np.array(img_resized)
    # Normalize the pixel values to be between -1 and 1
    img_normalized = (img_resized_array / 255 - 0.5) / 0.5
    # Add batch and channel dimensions to match the model's input shape
    img_tensor = torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0).float()
    
    # Get predictions from the model
    with torch.no_grad():
        outputs = net(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
    
    # Update the progress bars with the predicted probabilities
    for i in range(10):
        bars[i].configure(value=probabilities[i]*100)
    
    # Print the probabilities for debugging purposes
    print(probabilities)

# Button to reset the canvas
def reset_canvas():
    canvas.delete('all')  # Clear the canvas
    draw.rectangle((0, 0, canvas_size, canvas_size), fill='white')  # Fill the image with white
    for bar in bars:
        bar.configure(value=0)  # Reset all progress bars

reset_button = ttk.Button(root, text="Reset Canvas", command=reset_canvas)
reset_button.grid(row=3, column=0, pady=10)

# Probability display
prob_frame = ttk.Frame(root)
prob_frame.grid(row=0, column=1, sticky='n')

labels = []
bars = []

for i in range(10):
    label = ttk.Label(prob_frame, text=f"Number {i}:")
    label.pack(anchor='w')
    bar = ttk.Progressbar(prob_frame, orient='horizontal', length=200, mode='determinate')
    bar.pack(fill='x', padx=5)
    labels.append(label)
    bars.append(bar)

# Button to display the processed image
def show_processed_image():
    img_array = np.array(img)
    img_inverted = 255 - img_array
    img_resized = Image.fromarray(img_inverted).resize((28, 28), resample=Image.BICUBIC)
    img_resized.show()  # Open the processed image in a new window

show_image_button = ttk.Button(root, text="Show Processed Image", command=show_processed_image)
show_image_button.grid(row=4, column=0, pady=10)

# Run the application
root.mainloop()