from ..module import Module
import numpy as np


class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute softmax activation function.
        
        Args:
            x: Input array (logits). Expected shape: (num_classes, batch_size)
            
        Returns:
            Softmax probabilities with same shape as input
        """
        # Numerical stability: subtract max before exp
        x_stable = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x_stable)
        self.output = exp_x / np.sum(exp_x, axis=0, keepdims=True)
        return self.output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Compute gradient of softmax with respect to input.
        
        Args:
            output_grad: Gradient flowing back from next layer
            
        Returns:
            Gradient with respect to input logits
        """
        # Softmax backward: for each element, we need Jacobian matrix
        # For numerical efficiency, we use the vectorized form
        # grad = softmax * (output_grad - sum(output_grad * softmax))
        sum_grad = np.sum(output_grad * self.output, axis=0, keepdims=True)
        return self.output * (output_grad - sum_grad)

