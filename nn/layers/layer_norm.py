import numpy as np
from ..module import Module


class LayerNorm(Module):
    """Layer Normalization module."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones((d_model, 1))  # Scale parameter
        self.beta = np.zeros((d_model, 1))  # Shift parameter
        self._params = {"gamma": self.gamma, "beta": self.beta}
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)
        self.input = None
        self.output = None
        self.mean = None
        self.var = None
        self.normalized = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of layer normalization.
        
        Args:
            x: Input of shape (d_model, seq_len) or (d_model, batch_size, seq_len)
        
        Returns:
            Normalized output with same shape as input
        """
        self.input = x
        # Compute mean and variance along the feature dimension (axis=0)
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.var = np.var(x, axis=0, keepdims=True)
        
        # Normalize
        self.normalized = (x - self.mean) / np.sqrt(self.var + self.eps)
        
        # Scale and shift
        self.output = self.gamma * self.normalized + self.beta
        return self.output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of layer normalization.
        
        Args:
            output_grad: Gradient from next layer, same shape as output
        
        Returns:
            Gradient with respect to input
        """
        # Parameter gradients
        self.gamma_grad = np.sum(output_grad * self.normalized, axis=1, keepdims=True)
        self.beta_grad = np.sum(output_grad, axis=1, keepdims=True)
        
        # Gradient through scale
        dnormalized = output_grad * self.gamma
        
        # Gradient through normalization
        # Standard layer norm backward formula
        N = self.input.shape[0]  # Number of features
        std = np.sqrt(self.var + self.eps)
        
        # Gradient through variance
        dvar = np.sum(dnormalized * (self.input - self.mean) * -0.5 * (self.var + self.eps) ** (-3/2), axis=0, keepdims=True)
        
        # Gradient through mean
        dmean = np.sum(dnormalized * -1 / std, axis=0, keepdims=True) + dvar * np.mean(-2 * (self.input - self.mean), axis=0, keepdims=True)
        
        # Gradient through input
        dx = dnormalized / std + dvar * 2 * (self.input - self.mean) / N + dmean / N
        
        return dx
    
    def update_params(self, learning_rate: float = 0.001):
        self.gamma -= learning_rate * self.gamma_grad
        self.beta -= learning_rate * self.beta_grad