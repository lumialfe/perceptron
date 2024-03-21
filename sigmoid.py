import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input datasets
inputs = np.array([[0,0],
                   [0,1],
                   [1,0],
                   [1,1]])
# Output dataset
outputs = np.array([[0],[1],[1],[0]])

# Initialize weights randomly with mean 0
weights0 = 2 * np.random.random((2, 4)) - 1
weights1 = 2 * np.random.random((4, 1)) - 1

# Training step
for i in range(10000):

    # Forward propagation
    input_layer = inputs
    hidden_layer = sigmoid(np.dot(input_layer, weights0))
    output_layer = sigmoid(np.dot(hidden_layer, weights1))

    # Backpropagation
    output_layer_error = outputs - output_layer
    output_layer_delta = output_layer_error * sigmoid_derivative(output_layer)
    
    hidden_layer_error = output_layer_delta.dot(weights1.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer)

    # Updating weights
    weights1 += hidden_layer.T.dot(output_layer_delta)
    weights0 += input_layer.T.dot(hidden_layer_delta)

# Test the network for a new input
new_input = np.array([1, 0]) # Example input
hidden_layer = sigmoid(np.dot(new_input, weights0))
output_layer = sigmoid(np.dot(hidden_layer, weights1))
print(output_layer)
