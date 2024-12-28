from __future__ import annotations
import random

class Layer:
    def __init__(self, size: int, activationFunction: function, acrivationDerivative: function):
        self.nodes: list[float] = []    # nodes[i] = i-th node value
        self.weights: list[list[float]] = []    # weights[i][j] = j-th weight of the i-th node value (j-th weight = j-th node of the next layer)
        self.bias: list[float] = []    # nodes[i] = i-th node bias
        
        self.costGradientW: list[list[float]] = []    # costGradientW[i][j] = gradient of the cost function with respect to the j-th weight of the i-th node
        self.costGradientB: list[float] = []    # costGradientB[i] = gradient of the cost function with respect to the bias of the i-th node
        
        # Initialize the nodes, weights and bias
        for _ in range(size):
            self.nodes.append(0)
            self.weights.append([])
            self.bias.append(random.uniform(-1, 1))
            self.costGradientW.append([])
            self.costGradientB.append(0)
        
        self.size = size
        self.activationFunction = activationFunction
        self.activationDerivative = acrivationDerivative
    
    # Create the weights of this layer
    def connectToNextLayer(self, nextLayer: Layer):
        for i in range(self.size):
            for _ in range(nextLayer.size):
                self.weights[i].append(random.uniform(-1, 1))
                self.costGradientW[i].append(0)
    
    def applyGradients(self, learningRate: float):
        for i in range(self.size):
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= self.costGradientW[i][j] * learningRate
            self.bias[i] -= self.costGradientB[i] * learningRate
        
        """if self.size == 3:
            print(self.weights[0][0])"""