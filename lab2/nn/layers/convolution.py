from ..module import Module
from ..utils import he_initialization, init_biases
import numpy as np
from typing import Tuple, Union


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]], 
                 stride: int = 1, padding: int = 0):
        super().__init__()
        
        if isinstance(kernel_size, int):
            self.kernel_height = self.kernel_width = kernel_size
        else:
            self.kernel_height, self.kernel_width = kernel_size
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        
        # Weight shape: (out_channels, in_channels, kernel_height, kernel_width)
        fan_in = in_channels * self.kernel_height * self.kernel_width
        fan_out = out_channels * self.kernel_height * self.kernel_width
        limit = np.sqrt(6. / (fan_in + fan_out))
        
        self.W = np.random.uniform(-limit, limit, 
                                  (out_channels, in_channels, self.kernel_height, self.kernel_width))
        self.bias = np.zeros((out_channels, 1))
        
        self._params = {"weights": self.W, "bias": self.bias}
        self.W_grad = np.zeros_like(self.W)
        self.bias_grad = np.zeros_like(self.bias)
        
        self.input = None
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        batch_size, in_channels, in_height, in_width = x.shape
        
        out_height = (in_height + 2 * self.padding - self.kernel_height) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_width) // self.stride + 1
        
        if self.padding > 0:
            padded_input = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                                 mode='constant', constant_values=0)
        else:
            padded_input = x
            
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for i in range(out_height):
                        for j in range(out_width):
                            h_start = i * self.stride
                            h_end = h_start + self.kernel_height
                            w_start = j * self.stride
                            w_end = w_start + self.kernel_width
                            
                            output[b, oc, i, j] += np.sum(
                                padded_input[b, ic, h_start:h_end, w_start:w_end] * 
                                self.W[oc, ic, :, :]
                            )
                output[b, oc, :, :] += self.bias[oc, 0]
                
        self.output = output
        return output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        batch_size, in_channels, in_height, in_width = self.input.shape
        _, out_channels, out_height, out_width = output_grad.shape
        
        self.W_grad = np.zeros_like(self.W)
        self.bias_grad = np.zeros_like(self.bias)
        input_grad = np.zeros_like(self.input)
        
        if self.padding > 0:
            padded_input = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                                 mode='constant', constant_values=0)
            padded_input_grad = np.zeros_like(padded_input, dtype=np.float64)
        else:
            padded_input = self.input
            padded_input_grad = input_grad.astype(np.float64)
            
        for b in range(batch_size):
            for oc in range(out_channels):
                self.bias_grad[oc, 0] += np.sum(output_grad[b, oc, :, :])
                
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_height
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_width
                        
                        for ic in range(in_channels):
                            self.W_grad[oc, ic, :, :] += (
                                padded_input[b, ic, h_start:h_end, w_start:w_end] * 
                                output_grad[b, oc, i, j]
                            )
                            
                            padded_input_grad[b, ic, h_start:h_end, w_start:w_end] += (
                                self.W[oc, ic, :, :] * 
                                output_grad[b, oc, i, j]
                            )
                            
        if self.padding > 0:
            input_grad = padded_input_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]
            
        return input_grad
    
    def update_params(self, learning_rate: float = 0.001):
        self.W -= learning_rate * self.W_grad
        self.bias -= learning_rate * self.bias_grad