from layer import Layer
import functions.activation_functions as activation_functions
import functions.cost_functions as cost_functions

class NeuralNetwork:
    def __init__(self, costFunction: str = "squaredError"):
        self.layers: list[Layer] = []
        # To know more about activation functions see https://en.wikipedia.org/wiki/Activation_function#Table_of_activation_functions
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
            "squaredError": cost_functions.squaredError
        }
        self.costFunctionsDerivatives = {
            "squaredError": cost_functions.squaredErrorDerivative
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
            self.layers[0].nodes[inputNodeIdx] = self.layers[0].activationFunction(inputs[inputNodeIdx])
        
        for layerIdx in range(1, len(self.layers)):
            prevLayer = self.layers[layerIdx - 1]
            for nodeIdx in range(self.layers[layerIdx].size):
                val = sum(prevLayer.nodes[i] * prevLayer.weights[i][nodeIdx] for i in range(prevLayer.size)) + self.layers[layerIdx].bias[nodeIdx]
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
        h = 0.1
        startCost = self.cost(xTrain, YTrain)
        
        # Calculate the gradient of the cost function with respect to the weights and biases (exluding the output layer)
        for layerIdx in range(len(self.layers) - 1):
            # Cost gradient with respect to the weights
            for nodeIdx in range(self.layers[layerIdx].size):
                for j in range(self.layers[layerIdx + 1].size):
                    self.layers[layerIdx].weights[nodeIdx][j] += h
                    desltaCost = self.cost(xTrain, YTrain) - startCost
                    self.layers[layerIdx].weights[nodeIdx][j] -= h
                    self.layers[layerIdx].costGradientW[nodeIdx][j] = desltaCost / h
            
            # Cost gradient with respect to the biases
            for biasIdx in range(self.layers[layerIdx].size):
                self.layers[layerIdx].bias[biasIdx] += h
                desltaCost = self.cost(xTrain, YTrain) - startCost
                self.layers[layerIdx].bias[biasIdx] -= h
                self.layers[layerIdx].costGradientB[biasIdx] = desltaCost / h
            
            self.layers[layerIdx].applyGradients(learningRate)
        
        # Cost gradient with respect to the biases of the output layer
        for biasIdx in range(self.layers[-1].size):
                self.layers[-1].bias[biasIdx] += h
                desltaCost = self.cost(xTrain, YTrain) - startCost
                self.layers[-1].bias[biasIdx] -= h
                self.layers[-1].costGradientB[biasIdx] = desltaCost / h
        
        self.layers[-1].applyGradients(learningRate)
    
    # Train the neural network
    def train(self, inputs: list[list[float]], expectedOutputs: list[list[float]], learningRate: float):
        for i in range(len(inputs)):
            self.learn(inputs[i], expectedOutputs[i], learningRate)
    
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
    nn.addLayer(1, "sigmoid")
    #print(nn.forward([0, 1]))
    
    
    x = [[0, 1]]
    Y = [[1]]
    while True:
        nn.train(x, Y, 0.01)
        print(nn.forward(x[0]))