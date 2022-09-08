import matplotlib.pyplot as plt
import numpy as np


# Normalize data
def normalize_data(x):
    return (x - np.mean(x))/np.std(x)


# Cost function
def cost_function(x, y, beta, bias):
    cost = (1/len(y)) * np.sum(np.square(y - (beta * x + bias)))
    return cost


# Weight updater
def update_weights(x, y, beta, bias, learning_rate):
    beta_grad = (1 / len(y)) * np.sum(-2 * x * (y - (beta*x + bias)))
    bias_grad = (1 / len(y)) * np.sum(-2 * (y - (beta*x + bias)))

    beta -= learning_rate*beta_grad
    bias -= learning_rate*bias_grad
    return beta, bias


def lin_reg(x, y, iter, plot_flag, verbose):
    # Normalize Data
    x_norm = normalize_data(x)
    y_norm = normalize_data(y)

    # Set up parameters
    cost_history = []
    beta, bias = 1, 1

    # Iterate and perform learning of weights
    for i in range(iter):
        # Run gradient descent and calculate new weights
        beta, bias = update_weights(x_norm, y_norm, beta, bias, 0.2)

        # Estimate new cost
        cost = cost_function(x_norm, y_norm, beta, bias)

        if i % 1 == 0 and verbose:
            print(f'Iteration: {i}:: Cost: {cost}, Beta: {beta}, Bias: {bias}')

    # Plot
    if plot_flag:
        plt.scatter(x_norm, y_norm)
        plt.plot(x_norm, beta*x_norm + bias)
        plt.show()
    return bias, beta


# Testing out code
X = np.random.random(25)
Y = X + np.random.random(25)

lin_reg(X, Y, iter=100, plot_flag=1, verbose=1)