import numpy as np
from ..module import Module
from ..layers.softmax import Softmax


class CrossEntropy(Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
        self.softmax_output = None
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute cross-entropy loss.
        
        Args:
            y_pred: Predicted logits. Shape: (num_classes, batch_size)
            y_true: True labels (one-hot encoded). Shape: (num_classes, batch_size)
            
        Returns:
            Loss values per sample. Shape: (batch_size,)
        """
        # Apply softmax to predictions
        self.softmax_output = self.softmax.forward(y_pred)
        
        # Clip to avoid log(0)
        epsilon = 1e-15
        softmax_pred = np.clip(self.softmax_output, epsilon, 1 - epsilon)
        
        # Compute cross-entropy loss: -sum(y_true * log(softmax_pred))
        return -np.sum(y_true * np.log(softmax_pred), axis=0)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cross-entropy loss with respect to logits.
        
        Args:
            y_pred: Predicted logits. Shape: (num_classes, batch_size)
            y_true: True labels (one-hot encoded). Shape: (num_classes, batch_size)
            
        Returns:
            Gradient with respect to logits. Shape: (num_classes, batch_size)
        """
        # Get softmax probabilities (compute again if not stored, or use stored value)
        if self.softmax_output is None:
            softmax_pred = self.softmax.forward(y_pred)
        else:
            softmax_pred = self.softmax_output
        
        # The gradient of cross-entropy loss w.r.t. logits simplifies to:
        # grad = softmax(logits) - y_true
        return softmax_pred - y_true