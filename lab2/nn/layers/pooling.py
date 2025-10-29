from ..module import Module
import numpy as np
from typing import Tuple, Union


class MaxPool2d(Module):
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], stride: int = None, padding: int = 0):
        super().__init__()
        
        # Handle kernel size
        if isinstance(kernel_size, int):
            self.kernel_height = self.kernel_width = kernel_size
        else:
            self.kernel_height, self.kernel_width = kernel_size
            
        # Handle stride
        if stride is None:
            self.stride = self.kernel_height  # Default to kernel size
        else:
            self.stride = stride
            
        self.padding = padding
        
        # For backward pass
        self.input = None
        self.output = None
        self.max_indices = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        batch_size, channels, in_height, in_width = x.shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_height) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_width) // self.stride + 1
        
        # Apply padding if needed
        if self.padding > 0:
            padded_input = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                                 mode='constant', constant_values=-np.inf)
        else:
            padded_input = x
            
        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))
        # Store indices for backward pass
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        # Perform max pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Calculate receptive field
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_height
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_width
                        
                        # Extract patch and find maximum
                        patch = padded_input[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)
                        output[b, c, i, j] = max_val
                        
                        # Find indices of maximum value
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        self.max_indices[b, c, i, j, 0] = h_start + max_idx[0]
                        self.max_indices[b, c, i, j, 1] = w_start + max_idx[1]
                        
        self.output = output
        return output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        batch_size, channels, in_height, in_width = self.input.shape
        _, _, out_height, out_width = output_grad.shape
        
        # Initialize input gradient
        input_grad = np.zeros_like(self.input)
        
        # Distribute gradients to the positions of maximum values
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Get the indices of the maximum value from forward pass
                        h_idx, w_idx = self.max_indices[b, c, i, j]
                        # Remove padding offset if padding was applied
                        if self.padding > 0:
                            h_idx -= self.padding
                            w_idx -= self.padding
                            
                        # Check bounds after removing padding
                        if 0 <= h_idx < in_height and 0 <= w_idx < in_width:
                            input_grad[b, c, h_idx, w_idx] += output_grad[b, c, i, j]
                        
        return input_grad


class AvgPool2d(Module):
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], stride: int = None, padding: int = 0):
        super().__init__()
        
        # Handle kernel size
        if isinstance(kernel_size, int):
            self.kernel_height = self.kernel_width = kernel_size
        else:
            self.kernel_height, self.kernel_width = kernel_size
            
        # Handle stride
        if stride is None:
            self.stride = self.kernel_height  # Default to kernel size
        else:
            self.stride = stride
            
        self.padding = padding
        
        # For backward pass
        self.input = None
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        batch_size, channels, in_height, in_width = x.shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_height) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_width) // self.stride + 1
        
        # Apply padding if needed
        if self.padding > 0:
            padded_input = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                                 mode='constant', constant_values=0)
        else:
            padded_input = x
            
        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # Perform average pooling
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Calculate receptive field
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_height
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_width
                        
                        # Compute average
                        patch = padded_input[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.mean(patch)
                        
        self.output = output
        return output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        batch_size, channels, in_height, in_width = self.input.shape
        _, _, out_height, out_width = output_grad.shape
        
        # Initialize input gradient
        input_grad = np.zeros_like(self.input)
        
        # Apply padding if needed
        if self.padding > 0:
            padded_input_grad = np.zeros((batch_size, channels, in_height + 2*self.padding, in_width + 2*self.padding))
        else:
            padded_input_grad = input_grad
            
        # Distribute gradients evenly to all positions in the pooling window
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        # Calculate receptive field
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_height
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_width
                        
                        # Distribute gradient evenly
                        grad_val = output_grad[b, c, i, j] / (self.kernel_height * self.kernel_width)
                        padded_input_grad[b, c, h_start:h_end, w_start:w_end] += grad_val
                        
        # Remove padding from input gradients if needed
        if self.padding > 0:
            input_grad = padded_input_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]
            
        return input_grad