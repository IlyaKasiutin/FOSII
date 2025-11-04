from ..module import Module
import numpy as np


class Tanh(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.tanh(x)
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        tanh_grad = 1 - np.square(np.tanh(self.output))
        return output_grad * tanh_grad
