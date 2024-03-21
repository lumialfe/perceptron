import numpy as np
import matplotlib.pyplot as plt
import parameters as p

# parameters
w = [] # weights
b = 0 # bias

# data
F1s = []


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

        val = validate(x, y)
        print('TRAIN: Epoch:', epoch, 'F1:', val, 'Acc:', val[1])
        F1s.append(val)
    return w, b

def validate(x, y):

    TP, TN, FP, FN = 0, 0, 0, 0

    n = x.shape[0]
    correct = 0
    for i in range(n):
        y_pred = perceptron(x[i])
        if y_pred == y[i]:
            correct += 1
            if y[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if y[i] == 1:
                FN += 1
            else:
                FP += 1

    F1 = 2 * TP / (2 * TP + FP + FN)
    return F1, correct / n

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
            y[i] = y[i] * noise

    # add outlier
    for i in range(n_outlier):
        y[i] = 1 - y[i]

    return x, y

def main():
    global w, b
    # generate data
    x, y = generate_data(p.seed, p.n, p.noise, 0)
    x2, y2 = generate_data(p.seed + 69, p.n, p.noise, 0)
    # train
    train(x, y, p.e, p.lr)
    # validate
    F1 = validate(x2, y2)
    print('VALID: F1:', F1)

    # plot
    plt.figure()
    # plt.plot(x[y == 0, 0], x[y == 0, 1], 'ro')
    # plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo')
    plt.plot(x2[y2 == 0, 0], x2[y2 == 0, 1], 'g+')
    plt.plot(x2[y2 == 1, 0], x2[y2 == 1, 1], 'y+')
    # plot decision boundary
    x1 = np.linspace(-3, 3, 100)
    x2 = -w[0] / w[1] * x1 - b / w[1]
    plt.plot(x1, x2, 'k-')
    plt.title('F1: ' + str(F1))
    plt.show()


if __name__ == '__main__':
    main()