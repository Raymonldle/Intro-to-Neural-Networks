import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def init(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = np.dot(self.weights, inputs) + self.biasreturn 
        sigmoid(total)

    weights = np.array([0,1])
    bias = 4
    n = Neuron(weights, bias)
    x = np.array([2,3])
    print(n.feedforward(x))