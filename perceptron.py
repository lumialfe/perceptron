import numpy as np
import matplotlib.pyplot as plt

# parameters
w = [] # weights
b = 0 # bias
e = 500 # number of epochs
lr = 0.01 # learning rate


def perceptron(x):
    global w, b
    v = np.dot(w, x) + b
    y = 1 if v > 0 else 0
    return y

def train(x, y, epochs=100, learning_rate=0.1):
    global w, b
    n, d = x.shape
    w = np.zeros(d)
    b = 0
    for epoch in range(epochs):
        for i in range(n):
            y_pred = perceptron(x[i])
            if y_pred != y[i]:
                w = w + learning_rate * (y[i] - y_pred) * x[i]
                b = b + learning_rate * (y[i] - y_pred)

        print('Epoch:', epoch, 'Accuracy:', validate(x, y))
    return w, b

def validate(x, y):
    n = x.shape[0]
    correct = 0
    for i in range(n):
        y_pred = perceptron(x[i])
        if y_pred == y[i]:
            correct += 1
    return correct / n

# generate random data based on the given seed and number of data points
# the data is plotted and returned
def generate_data(seed=0, n=100, noise=0.1, n_outlier=0):
    np.random.seed(seed)
    x = np.random.randn(n, 2)
    w0 = 2
    w1 = 3
    b = -2
    y = np.zeros(n)
    for i in range(n):
        if w0 * x[i, 0] + w1 * x[i, 1] + b > 0:
            y[i] = 1

    # add noise
    for i in range(n):
        if np.random.rand() < noise:
            y[i] = 1 - y[i]

    # add outlier
    for i in range(n_outlier):
        y[i] = 1 - y[i]

    return x, y

def main():
    global w, b
    # generate data
    x, y = generate_data()
    x2, y2 = generate_data(seed=1)
    # train
    train(x, y, e, lr)
    # validate
    acc = validate(x2, y2)
    print('Accuracy:', acc)

if __name__ == '__main__':
    main()