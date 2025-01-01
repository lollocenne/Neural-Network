# Read the data from the MNIST database https://yann.lecun.com/exdb/mnist/


import numpy as np
import struct

def readTrainingImages(filePath):
    with open(filePath, 'rb') as f:
        magicNumber, numImages, numRows, numCols = struct.unpack('>IIII', f.read(16))
        assert magicNumber == 2051, "Not valid file format"
        images = np.fromfile(f, dtype=np.uint8).reshape(numImages, numRows, numCols)
    return images

def readTrainingLabels(filePath):
    with open(filePath, 'rb') as f:
        magicNumber = struct.unpack('>II', f.read(8))[0]
        assert magicNumber == 2049, "Not valid file format"
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

testPath = "test/training_data/"
imagesFilePath = testPath + "train-images.idx3-ubyte"
labelsFilePath = testPath + "train-labels.idx1-ubyte"

images = readTrainingImages(imagesFilePath)
labels = readTrainingLabels(labelsFilePath)

def getTrainingData():
    return (images, labels)


if __name__ == "__main__":
    print(f"Immagini: {images.shape}")
    print(f"Etichette: {labels.shape}")
    
    for i in range(5):
        print(f"Immagine {i+1}: Etichetta = {labels[i]}")
