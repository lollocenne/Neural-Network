import random

class Layer:
    BETA = 0.9
    def __init__(self, size: int, activationFunction: "function", acrivationDerivative: "function"):
        self.nodes: list[float] = []    # nodes[i] = i-th node value
        self.weights: list[list[float]] = []    # weights[i][j] = j-th weight of the i-th node value (j-th weight = j-th node of the next layer)
        self.bias: list[float] = []    # nodes[i] = i-th node bias
        
        self.costGradientW: list[list[float]] = []    # costGradientW[i][j] = gradient of the cost function with respect to the j-th weight of the i-th node
        self.costGradientB: list[float] = []    # costGradientB[i] = gradient of the cost function with respect to the bias of the i-th node
        
        self.momentumW: list[list[float]] = []  # momentumW[i][j] = momentum of the weights[i][j]
        self.momentumB: list[float] = []  # momentumB[i] = momentum of the bias[i]
        
        self.weightedVal: list[float] = []    # weightedVal[i] = i-th node value without the activation function
        self.actCos: list[float] = []    # actCos[i] = derivative of the activation function times the derivative of cost function of the i-th node
        
        # Initialize the nodes, weights and bias
        for _ in range(size):
            self.nodes.append(0)
            self.weights.append([])
            self.bias.append(random.uniform(-1, 1))
            self.costGradientW.append([])
            self.costGradientB.append(0)
            self.momentumW.append([])
            self.momentumB.append(0)
            self.weightedVal.append(0)
            self.actCos.append(0)
        
        self.size = size
        self.activationFunction = activationFunction
        self.activationDerivative = acrivationDerivative
    
    # Create the weights of this layer
    def connectToNextLayer(self, nextLayer: "Layer") -> None:
        for i in range(self.size):
            for _ in range(nextLayer.size):
                self.weights[i].append(random.uniform(-1, 1))
                self.costGradientW[i].append(0)
                self.momentumW[i].append(0)
    
    # Apply the gradients to the weights and bias
    # Reset the gradients to 0 to prepare them for the next training
    def applyGradients(self, learningRate: float, batchSize: int) -> None:
        for i in range(self.size):
            for j in range(len(self.weights[i])):
                self.costGradientW[i][j] /= batchSize
                self.momentumW[i][j] = Layer.BETA * self.momentumW[i][j] + (1 - Layer.BETA) * self.costGradientW[i][j]
                self.weights[i][j] -= self.momentumW[i][j] * learningRate
                self.costGradientW[i][j] = 0
            self.momentumB[i] /= batchSize
            self.momentumB[i] = Layer.BETA * self.momentumB[i] + (1 - Layer.BETA) * self.costGradientB[i]
            self.bias[i] -= self.momentumB[i] * learningRate
            self.costGradientB[i] = 0
    
    # Calculate the derivative of the cost function times the derivative of the activation function of the output layer
    def calculateOutputLayerActCos(self, expectedOutputs: list[float], costDerivative: "function") -> list[float]:
        actCoss: list[float] = [0 for _ in range(len(expectedOutputs))]
        for i in range(len(expectedOutputs)):
            activationDerivative = self.activationDerivative(self.weightedVal[i])
            costDerivativeVal = costDerivative(self.nodes[i], expectedOutputs[i])
            actCoss[i] = activationDerivative * costDerivativeVal
        return actCoss
    
    # Calculate the derivative of the cost function times the derivative of the activation function of the hidden layers
    def calculateHiddenLayerActCos(self, nextActCos: list[float], nextLayer: "Layer") -> list[float]:
        actCoss: list[float] = []
        for nodeIdx in range(self.size):
            actCos = 0
            for nextNodeIdx in range(nextLayer.size):
                weightedInputDerivative = self.weights[nodeIdx][nextNodeIdx]
                actCos += weightedInputDerivative * nextActCos[nextNodeIdx]
            actCos *= self.activationDerivative(self.weightedVal[nodeIdx])
            actCoss.append(actCos)
        return actCoss
    
    # Update the gradients of the layer except the input layer
    def updateGradients(self, actCos: list[float], prevLayer: "Layer") -> None:
        for nodeIdx in range(self.size):
            for prevNodeIdx in range(prevLayer.size):
                derivativeCostWeight = actCos[nodeIdx] * prevLayer.nodes[prevNodeIdx]
                prevLayer.costGradientW[prevNodeIdx][nodeIdx] += derivativeCostWeight
            derivativeCostBias = actCos[nodeIdx]
            self.costGradientB[nodeIdx] += derivativeCostBias
    
    # Update the gradients of the input layer
    def updateInputLayerGradients(self, actCos: list[float]) -> None:            
        for nodeIdx in range(self.size):
            derivativeCostBias = actCos[nodeIdx]
            self.costGradientB[nodeIdx] += derivativeCostBias