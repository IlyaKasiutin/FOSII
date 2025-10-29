import numpy as np
from ..module import Module


class CrossEntropy(Module):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Forward pass of Cross Entropy loss
        
        Args:
            y_pred: Predicted values (logits) of shape (n_classes, batch_size)
            y_true: True values (one-hot encoded) of shape (n_classes, batch_size)
            
        Returns:
            Loss values of shape (batch_size,)
        """
        # Apply softmax to convert logits to probabilities
        # Subtract max for numerical stability
        y_pred_stable = y_pred - np.max(y_pred, axis=0, keepdims=True)
        exp_pred = np.exp(y_pred_stable)
        softmax_pred = exp_pred / np.sum(exp_pred, axis=0, keepdims=True)
        
        # Compute cross entropy loss
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        softmax_pred = np.clip(softmax_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(softmax_pred), axis=0)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Backward pass of Cross Entropy loss
        
        Args:
            y_pred: Predicted values (logits) of shape (n_classes, batch_size)
            y_true: True values (one-hot encoded) of shape (n_classes, batch_size)
            
        Returns:
            Gradient of shape (n_classes, batch_size)
        """
        # Apply softmax to convert logits to probabilities
        # Subtract max for numerical stability
        y_pred_stable = y_pred - np.max(y_pred, axis=0, keepdims=True)
        exp_pred = np.exp(y_pred_stable)
        softmax_pred = exp_pred / np.sum(exp_pred, axis=0, keepdims=True)
        
        # Gradient of cross entropy loss with softmax is (pred - true)
        return softmax_pred - y_true