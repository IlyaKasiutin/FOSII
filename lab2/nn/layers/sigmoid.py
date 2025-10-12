from ..module import Module
from ..tensor import Tensor
import numpy as np


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x: Tensor) -> Tensor:
        self.output =  1 / (1 + np.exp(-x.data))
        return Tensor(self.output, requires_grad=True)
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_grad = self.output * (1 - self.output)
        return output_grad * sigmoid_grad
    