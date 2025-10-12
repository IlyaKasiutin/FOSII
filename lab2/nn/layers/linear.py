from ..module import Module
from ..tensor import Tensor
from utils import he_initialization, init_biases
import numpy as np


class Linear(Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        super().__init__()
        self.W = he_initialization(n_inputs, n_outputs)
        self.bias = init_biases(n_outputs)
        self.output = None

    def forward(self, x: Tensor) -> Tensor:
        self.output = self.W @ x.data.T  # (n_out, n_in) * (n_in, 1) = (n_out, 1)
        self.output = self.output.T  # (1, n_out)
        return Tensor(self.output, requires_grad=True)
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        sigmoid_grad = self.output * (1 - self.output)
        return output_grad * sigmoid_grad
    