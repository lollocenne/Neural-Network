from __future__ import annotations
import random

class Layer:
    def __init__(self, size: int, activationFunction: function, acrivationDerivative: function):
        self.nodes: list[float] = []    # nodes[i] = i-th node value
        self.weights: list[list[float]] = []    # weights[i][j] = j-th weight of the i-th node value (j-th weight = j-th node of the next layer)
        self.bias: list[float] = []    # nodes[i] = i-th node bias
        
        self.costGradientW: list[list[float]] = []    # costGradientW[i][j] = gradient of the cost function with respect to the j-th weight of the i-th node
        self.costGradientB: list[float] = []    # costGradientB[i] = gradient of the cost function with respect to the bias of the i-th node
        
        self.weightedVal = []    # weightedVal[i] = i-th node value without the activation function
        self.actCos = []    # actCos[i] = derivative of the activation function times the derivative of cost function of the i-th node
        
        # Initialize the nodes, weights and bias
        for _ in range(size):
            self.nodes.append(0)
            self.weights.append([])
            self.bias.append(random.uniform(-1, 1))
            self.costGradientW.append([])
            self.costGradientB.append(0)
            self.weightedVal.append(0)
            self.actCos.append(0)
        
        self.size = size
        self.activationFunction = activationFunction
        self.activationDerivative = acrivationDerivative
    
    # Create the weights of this layer
    def connectToNextLayer(self, nextLayer: Layer):
        for i in range(self.size):
            for _ in range(nextLayer.size):
                self.weights[i].append(random.uniform(-1, 1))
                self.costGradientW[i].append(0)
    
    # Apply the gradients to the weights and bias
    def applyGradients(self, learningRate: float):
        for i in range(self.size):
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= self.costGradientW[i][j] * learningRate
            self.bias[i] -= self.costGradientB[i] * learningRate
    
    # Calculate the derivative of the cost function times the derivative of the activation function of the output layer
    def calculateOutputLayerActCos(self, expectedOutputs: list[float], costDerivative: function) -> list[float]:
        actCoss: list[float] = [0 for _ in range(len(expectedOutputs))]
        for i in range(len(expectedOutputs)):
            activationDerivative = self.activationDerivative(self.weightedVal[i])
            costDerivative = costDerivative(self.nodes[i], expectedOutputs[i])
            actCoss[i] = activationDerivative * costDerivative
        return actCoss
    
    # Reset the gradients to 0
    def resetGradients(self):
        for i in range(self.size):
            for j in range(len(self.weights[i])):
                self.costGradientW[i][j] = 0
            self.costGradientB[i] = 0
    
    # Update the gradients of the layer except the input layer
    def updateGradients(self, actCos: list[float], prevLayer: Layer):
        for nodeIdx in range(self.size):
            for prevNodeIdx in range(prevLayer.size):
                derivativeCostWeight = actCos[nodeIdx] * prevLayer.nodes[prevNodeIdx]
                prevLayer.costGradientW[prevNodeIdx][nodeIdx] += derivativeCostWeight
            
            derivativeCostBias = actCos[nodeIdx]
            self.costGradientB[nodeIdx] += derivativeCostBias
    
    # Update the gradients of the input layer
    def updateInputLayerGradients(self, actCos: list[float]):            
        for nodeIdx in range(self.size):
            derivativeCostBias = actCos[nodeIdx]
            self.costGradientB[nodeIdx] += derivativeCostBias
    
    def calculateHiddenLayerActCos(self, nextActCos: list[float], nextLayer: Layer):
        actCoss: list[float] = []
        for nodeIdx in range(self.size):
            actCos = 0
            for nextNodeIdx in range(nextLayer.size):
                weightedInputDerivative = self.weights[nodeIdx][nextNodeIdx]
                actCos += weightedInputDerivative * nextActCos[nextNodeIdx]
            actCos *= self.activationDerivative(self.weightedVal[nodeIdx])
            actCoss.append(actCos)
        return actCoss