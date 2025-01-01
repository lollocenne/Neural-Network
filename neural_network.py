from layer import Layer
import functions.activation_functions as activation_functions
import functions.cost_functions as cost_functions

class NeuralNetwork:
    def __init__(self, costFunction: str = "squaredError"):
        self.layers: list[Layer] = []
        self.activationFunctions = {
            "binaryStep": activation_functions.binaryStep,
            "linear": activation_functions.linear,
            "sigmoid": activation_functions.sigmoid,
            "tanh": activation_functions.tanh,
            "relu": activation_functions.relu
        }
        self.activationFunctionsDerivatives = {
            "binaryStep": activation_functions.binaryStepDerivative,
            "linear": activation_functions.linearDerivative,
            "sigmoid": activation_functions.sigmoidDerivative,
            "tanh": activation_functions.tanhDerivative,
            "relu": activation_functions.reluDerivative
        }
        self.costFunctions = {
            "squaredError": cost_functions.squaredError,
            "crossEntropy": cost_functions.crossEntropy
        }
        self.costFunctionsDerivatives = {
            "squaredError": cost_functions.squaredErrorDerivative,
            "crossEntropy": cost_functions.crossEntropyDerivative
        }
        self.costFunction: function = self.costFunctions[costFunction]
        self.costDerivative: function = self.costFunctionsDerivatives[costFunction]
    
    # Add a layer to the neural network, connect the last layer to the new layer
    def addLayer(self, size: int, activationFunction: str = "linear"):
        self.layers.append(Layer(size, self.activationFunctions[activationFunction], self.activationFunctionsDerivatives[activationFunction]))
        if len(self.layers) > 1:
            self.layers[-2].connectToNextLayer(self.layers[-1])
    
    # Forward propagation
    def forward(self, inputs: list[float]) -> list[float]:
        for inputNodeIdx in range(self.layers[0].size):
            self.layers[0].weightedVal[inputNodeIdx] = inputs[inputNodeIdx]
            self.layers[0].nodes[inputNodeIdx] = self.layers[0].activationFunction(inputs[inputNodeIdx])
        
        for layerIdx in range(1, len(self.layers)):
            prevLayer = self.layers[layerIdx - 1]
            for nodeIdx in range(self.layers[layerIdx].size):
                val = sum(prevLayer.nodes[i] * prevLayer.weights[i][nodeIdx] for i in range(prevLayer.size))
                self.layers[layerIdx].weightedVal[nodeIdx] = val
                val += self.layers[layerIdx].bias[nodeIdx]
                self.layers[layerIdx].nodes[nodeIdx] = self.layers[layerIdx].activationFunction(val)
        
        return self.layers[-1].nodes
    
    # Calculate the cost of the neural network
    def cost(self, inputs: list[float], Y: list[float]) -> float:
        """
        inputs: input of the neural network
        Y: expected output (for heach node)
        """
        x: list[float] = self.forward(inputs)   # Output of the neural network
        cost: float = 0
        for i in range(self.layers[-1].size):
            cost += self.costFunction(x[i], Y[i])
        return cost
    
    # Learn using the gradient descent
    def learn(self, xTrain: list[float], YTrain: list[float], learningRate: float):
        self.updateGradients(xTrain, YTrain)
        for layer in self.layers:
            layer.applyGradients(learningRate)
    
    # Update the gradients of the neural network
    def updateGradients(self, inputs: list[float], Y: list[float]):
        self.forward(inputs)
        # Output layer
        actCos = self.layers[-1].calculateOutputLayerActCos(Y, self.costDerivative)   # Activation function derivative times cost function derivative
        self.layers[-1].updateGradients(actCos, self.layers[-2])
        # Hiddes layers
        for layerIdx in range(len(self.layers) - 2, 0, -1):
            actCos = self.layers[layerIdx].calculateHiddenLayerActCos(actCos, self.layers[layerIdx + 1])
            self.layers[layerIdx].updateGradients(actCos, self.layers[layerIdx - 1])
        # Input layer
        actCos = self.layers[0].calculateHiddenLayerActCos(actCos, self.layers[1])
        self.layers[0].updateInputLayerGradients(actCos)
    
    # Train the neural network
    def train(self, inputs: list[list[float]], expectedOutputs: list[list[float]], learningRate: float, batchSize: int = None, showProgress = False):
        if batchSize is None: batchSize = len(inputs)
        numBatches = (len(inputs) + batchSize - 1) // batchSize
        for batchIndex in range(numBatches):
            startIndex = batchIndex * batchSize
            endIndex = min(startIndex + batchSize, len(inputs))
            batchInputs = inputs[startIndex:endIndex]
            batchOutputs = expectedOutputs[startIndex:endIndex]
            for i in range(len(batchInputs)):
                self.learn(batchInputs[i], batchOutputs[i], learningRate)
                if showProgress:
                    print(f"Training: {round((batchIndex + i + 1) * 100 / len(inputs), 2)}%")
    
    
    def __str__(self):
        neuralNetwork = ""
        for layer in self.layers:
            for node in layer.nodes:
                neuralNetwork += f"{node} "
            neuralNetwork += "\n"
        return neuralNetwork


if __name__ == "__main__":
    nn = NeuralNetwork("squaredError")
    nn.addLayer(2)
    nn.addLayer(3, "tanh")
    nn.addLayer(1, "linear")
    
    
    x = [[2, 1]]
    Y = [[3]]
    while True:
        nn.train(x, Y, 0.0001)
        print(nn.forward(x[0]))