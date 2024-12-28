# Cost of single node

# Cost functions

def squaredError(x: float, Y: float) -> float:
        """
        x: output of the node
        Y: expected output
        """
        error = x - Y
        return 0.5 * error ** 2

# Cost functions derivatives

def squaredErrorDerivative(x: float, Y: float) -> float:
        """
        x: output of the node
        Y: expected output
        """
        return x - Y