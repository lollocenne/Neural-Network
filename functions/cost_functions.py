# Cost of single node

import math


EPSILON = 1e-15

# Cost functions

def squaredError(predicted: float, expected: float) -> float: return 0.5 * (predicted - expected) ** 2
def crossEntropy(predicted: float, expected: float) -> float:
        predicted = max(min(predicted, 1 - EPSILON), EPSILON)
        return - (expected * math.log(predicted) + (1 - expected) * math.log(1 - predicted))

# Cost functions derivatives

def squaredErrorDerivative(predicted: float, expected: float) -> float: return predicted - expected
def crossEntropyDerivative(predicted: float, expected: float) -> float:
        predicted = max(min(predicted, 1 - EPSILON), EPSILON)
        return - (expected / predicted) + ((1 - expected) / (1 - predicted))