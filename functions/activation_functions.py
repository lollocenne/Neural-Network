import math

# Activation functions

def binaryStep(x): return 1 if x > 0 else 0
def linear(x): return x
def sigmoid(x): return 1 / (1 + math.exp(-x))
def tanh(x): return math.tanh(x)
def relu(x): return max(0, x)

# Activation functions derivatives

def binaryStepDerivative(x): return 0
def linearDerivative(x): 1
def sigmoidDerivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)
def tanhDerivative(x): return 1 - tanh(x) ** 2
def reluDerivative(x): return 1 if x > 0 else 0