import sys
sys.path.append('../NEURAL-NETWORK')

# Preparing the data
from training_data.get_data import getTrainingData

trainingImages, trainingLabels = getTrainingData()

trainingImagesShaped = []   # Reshape the images to be a 784x1 instead of 28x28
trainingLabelsShaped = []   # For esample, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] instead of 5
for i in range(trainingImages.shape[0]):
    label = [0] * 10
    label[trainingLabels[i]] = 1
    trainingLabelsShaped.append(label)
    
    image = []
    for j in range(trainingImages.shape[1]):
        for k in range(trainingImages.shape[2]):
            image.append(int(trainingImages[i][j][k])/255)
    trainingImagesShaped.append(image)
    if (i + 1) % 3000 == 0:
        print(f"Data reshaped: {round(i * 100 / trainingImages.shape[0])}%")

# Splitting the data into training and testing
lenTest = 100
testImagesShaped, trainingImagesShaped = trainingImagesShaped[:lenTest], trainingImagesShaped[lenTest:]
testLabelsShaped, trainingLabelsShaped = trainingLabelsShaped[:lenTest], trainingLabelsShaped[lenTest:]


# Training the neural network
from neural_network import NeuralNetwork
# Return the answer
def getRes(output: list) -> int:
    max = 0
    num = -1
    for i, n in enumerate(output):
        if n > max:
            max = n
            num = i
    return num

# Return the accuracy of the neural network given multiple input
# Return both accuracy using training data and testing data
def getAccuracy(nn: NeuralNetwork) -> tuple[float, float]:
    totTest = totTrain = 0
    for i in range(lenTest):
        print(f"Test{i} : ", end="")
        
        resTest = getRes(nn.forward(testImagesShaped[i]))
        expTest = getRes(testLabelsShaped[i])
        print(resTest, expTest)
        if resTest == expTest:
            totTest += 1
        
        resTrain = getRes(nn.forward(trainingImagesShaped[i]))
        expTrain = getRes(trainingLabelsShaped[i])
        if resTrain == expTrain:
            totTrain += 1
    
    trainAccuracy = totTrain * 100 / lenTest
    testAccuracy = totTest * 100 / lenTest
    return (trainAccuracy, testAccuracy)

nn = NeuralNetwork().getFromSave("digits")  # Load the neural network
# This neural network has a 94% accuracy

"""
# Create a neural network (not needed when already loading the neural network)
nn = NeuralNetwork("crossEntropy")
nn.addLayer(784)
nn.addLayer(128, "relu")
nn.addLayer(64, "relu")
nn.addLayer(10, "sigmoid")
"""

accuracy = (0, 0)
# Train the neural network
while accuracy[1] <= 97: # Stop when accuracy is more then 97%
    nn.train(trainingImagesShaped, trainingLabelsShaped, 0.001, batchSize=5, showProgress=True)
    nn.save("digits")
    accuracy = getAccuracy(nn)

# Show the accuracy and check for overfitting
accuracy = getAccuracy(nn)
print(f"Accuracy Training: {accuracy[0]}%")    # Accoracy for recognizing the training data
print(f"Accuracy Test: {accuracy[1]}%")    # Accoracy for recognizing the test data