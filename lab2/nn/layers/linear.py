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
        self.input = x  # (n_in, batch) or (n_in, 1)
        self.output = self.W @ x + self.bias  # (n_out, n_in) * (n_in, batch) = (n_out, batch)
        return self.output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        # output_grad shape: (n_out, batch_size)
        # self.input shape: (n_in, batch_size)
        # W shape: (n_out, n_in)
        
        # Weight gradient: (n_out, batch) * (batch, n_in) = (n_out, n_in)
        self.W_grad = output_grad @ self.input.T
        
        # Bias gradient: (n_out, batch) -> (n_out, 1) by summing over batch dimension
        self.bias_grad = np.sum(output_grad, axis=1, keepdims=True)
        
        # Input gradient: (n_in, n_out) * (n_out, batch) = (n_in, batch)
        input_grad = self.W.T @ output_grad
        
        return input_grad
    
    def update_params(self, learning_rate: float = 0.001):
        self.W -= learning_rate * self.W_grad
        self.bias -= learning_rate * self.bias_grad