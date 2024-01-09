import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("heart.csv")
data = data.drop(columns = ["sex","restecg","thalach","exang","oldpeak","slope","thal","trestbps","chol"])
X = data[["age","cp","fbs","ca"]].to_numpy()
Y = data["target"].to_numpy()
Y = np.expand_dims(Y, axis=1)

input_size = 4
hidden_layer_size = 8
output_size = 1
learning_rate = 0.02
num_iterations = 1000

W1 = np.random.randn(input_size, hidden_layer_size)

b1 = np.zeros((1, hidden_layer_size))

W2 = np.random.randn(hidden_layer_size, output_size)

b2 = np.zeros((1, output_size))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_loss(Y, Y_hat):
    m = Y.shape[0]
    loss = -(1 / m) * np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return loss


costs = []

for i in range(num_iterations):

    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    Y_hat = sigmoid(Z2)

    loss = compute_loss(Y, Y_hat)
    costs.append(loss)

    dZ2 = Y_hat - Y
    dW2 = (1 / X.shape[0]) * np.dot(A1.T, dZ2)
    db2 = (1 / X.shape[0]) * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = (1 / X.shape[0]) * np.dot(X.T, dZ1)
    db1 = (1 / X.shape[0]) * np.sum(dZ1, axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if i % 100 == 0:
        print(f'Iteration {i}, Cost: {loss}')

plt.plot(costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations')
plt.show()