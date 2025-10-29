import numpy as np
from typing import List
from .module import Module
from .layers.linear import Linear
from .layers.sigmoid import Sigmoid
from .losses.mse import MSE


class MLP(Module):
    def __init__(self, dims: List[int]):
        """
        Multi-Layer Perceptron implementation in PyTorch style
        
        Args:
            dims: List of layer dimensions [input_dim, hidden_dim1, ..., output_dim]
        """
        super().__init__()
        self.layers = []
        
        # Create layers
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))
            # Add activation function except for the output layer
            if i < len(dims) - 2:
                self.layers.append(Sigmoid())
                
        self._modules = {f"layer_{i}": layer for i, layer in enumerate(self.layers)}
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the MLP
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Transpose to match layer expectations (features, batch)
        result = x.T
        
        # Forward through all layers
        for layer in self.layers:
            result = layer(result)
            
        # Transpose back to (batch, features)
        return result.T
    
    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Backward pass of the MLP
        
        Args:
            loss_grad: Gradient from loss function of shape (batch_size, output_dim)
        """
        # Transpose to match layer expectations (features, batch)
        output_grad = loss_grad  # (batch, features)
        
        # Backward through all layers in reverse order
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)
    
    def update_params(self, learning_rate: float = 0.001) -> None:
        """
        Update parameters of all layers
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        for layer in self.layers:
            if hasattr(layer, 'update_params'):
                layer.update_params(learning_rate)
    
    def zero_grad(self) -> None:
        """
        Zero out gradients of all layers
        """
        for layer in self.layers:
            if hasattr(layer, 'W_grad'):
                layer.W_grad.fill(0)
            if hasattr(layer, 'bias_grad'):
                layer.bias_grad.fill(0)