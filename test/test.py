import sys
sys.path.append('../NEURAL-NETWORK')

# Preparing the data
from training_data.get_data import getTrainingData

trainingImages, trainingLabels = getTrainingData()

trainingImagesShaped = []   # Reshape the images to be a 784x1 instead of 28x28
traininglabelsShaped = []   # For esample, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] instead of 5
for i in range(trainingImages.shape[0]):
    label = [0] * 10
    label[trainingLabels[i]] = 1
    traininglabelsShaped.append(label)
    
    image = []
    for j in range(trainingImages.shape[1]):
        for k in range(trainingImages.shape[2]):
            image.append(int(trainingImages[i][j][k])/255)
    trainingImagesShaped.append(image)
    if (i + 1) % 3000 == 0:
        print(f"Data reshaped: {round(i * 100 / trainingImages.shape[0])}%")


# Training the neural network
from neural_network import NeuralNetwork

nn = NeuralNetwork("crossEntropy")
nn.addLayer(784)
nn.addLayer(128, "relu")
nn.addLayer(64, "relu")
nn.addLayer(10, "sigmoid")

# It just needs 1 iteration to get enough accuracy 
for i in range(1):
    nn.train(trainingImagesShaped, traininglabelsShaped, 0.001, showProgress=True)