from ..module import Module
import numpy as np


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output =  1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_grad = self.output * (1 - self.output)
        return output_grad * sigmoid_grad
    