## Convolutional Neural Network
This project implements a custom Convolutional Neural Network (CNN) from scratch in Python, using fundamental libraries such as NumPy, SciPy, and TensorFlow (for dataset loading). It includes forward and backward propagation for convolutional, pooling, and fully connected layers, and applies these to the MNIST dataset for handwritten digit classification.

## Features
- **Convolution Layer:** Implements 2D convolution with learnable filters and biases.
- **MaxPooling Layer:** Performs max pooling for down-sampling with gradients computed for backpropagation.
- **Fully Connected Layer:** Includes a fully connected layer with softmax activation for classification.
- **Cross-Entropy Loss:** Calculates and propagates gradients for the loss.
- **Custom Training Loop:** Trains the CNN using manually implemented forward and backward passes.
- **MNIST Digit Classification:** Prepares the MNIST dataset for training and testing.
- **Interactive Input:** Accepts user-drawn digit images for prediction using a GUI built with Tkinter.

## Installation
**Clone the repository:**
```bash
git clone https://github.com/your-username/custom-cnn.git
cd custom-cnn
```
**Install the required dependencies:**
```bash
pip install numpy scipy matplotlib tensorflow pillow scikit-learn
```
**Usage**
- Open the uploaded Python notebook and run the code block one after another in the given sequence.
- Input the following parameters: Filter size for convolution, Number of filters, Pooling size.

The script will train the CNN on the MNIST dataset and display training metrics (loss and accuracy) for each epoch.

**Making Predictions**
Test Dataset: After training, the script evaluates the test set and displays the accuracy.

**Custom Input:** 
- Draw a digit using the Tkinter-based GUI by running the interactive input function.
- Save the drawing and use the prediction function to classify it.

# File Structure
- CNN_1.ipynb: Main Python script containing the CNN implementation and training logic.

# Dependencies
- Python 3.8+
- NumPy
- SciPy
- TensorFlow
- Matplotlib
- Pillow
- Scikit-learn
- Tkinter (included with Python)
