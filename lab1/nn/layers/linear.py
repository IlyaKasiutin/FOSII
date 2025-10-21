from ..module import Module
from ..utils import he_initialization, init_biases
import numpy as np


class Linear(Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        super().__init__()
        self.W = he_initialization(n_inputs, n_outputs)
        self.bias = init_biases(n_outputs)
        self._params = {"weights": self.W, "bias": self.bias}
        self.W_grad = np.zeros_like(self.W)
        self.bias_grad = np.zeros_like(self.bias)
        self.output = None
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x  # (1, n_in)
        self.output = self.W @ x + self.bias # (n_out, n_in) * (n_in, 1) = (n_out, 1)
        return self.output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        w_grad  = output_grad * self.input
        self.W_grad = w_grad
        self.b_grad = output_grad
        return output_grad * self.W
    
    def update_params(self):
        for param in self._params:
            
    