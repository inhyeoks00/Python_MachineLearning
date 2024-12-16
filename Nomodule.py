from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

class LogisticNeuron():
    def __init__(self):
        self.w = None
        self.b = None
        self.losses = []

    def forpass(self, x):
        z = np.sum(x * self.w) + self.b
        return z

    def backprop(self, x, err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad

    def activation(self, z, k):
        a = 1 / (1 + np.exp(-k * z))
        return a

    def loss(self, y, a):
        return -(y * np.log(a) + (1 - y) * np.log(1 - a))

    def fit(self, x, y, k, epochs=100):
        self.w = np.ones(x.shape[1])
        self.b = 0
        self.losses = []

        for i in range(epochs):
            epoch_loss = 0
            for x_i, y_i in zip(x, y):
                z = self.forpass(x_i)
                a = self.activation(z, k)
                err = -k * (y_i - a)
                w_grad, b_grad = self.backprop(x_i, err)
                self.w -= w_grad
                self.b -= b_grad

                a = np.clip(a, 1e-10, 1 - 1e-10)
                epoch_loss += self.loss(y_i, a)
            self.losses.append(epoch_loss / len(y))

    def predict(self, x, k):
        z = [self.forpass(x_i) for x_i in x]
        a = self.activation(np.array(z), k)
        return a > 0.5


cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)

k_values = np.linspace(0.1,100, 10)  
train_accuracies = []
test_accuracies = []

for k in k_values:
    neuron = LogisticNeuron()
    neuron.fit(x_train, y_train, k)
    
    train_accuracy = np.mean(neuron.predict(x_train, k) == y_train)
    test_accuracy = np.mean(neuron.predict(x_test, k) == y_test)
    
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)


plt.plot(k_values, test_accuracies, marker='o')
plt.xlabel("k value (regularization strength)")
plt.ylabel("Accuracy")
plt.title("Effect of k on Model Accuracy")
plt.legend()
plt.grid(True)
plt.show()
