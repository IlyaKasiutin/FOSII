import numpy as np
from ..module import Module
from typing import Union


class MSE(Module):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Forward pass of MSE loss
        
        Args:
            y_pred: Predicted values of shape (n_outputs, batch_size)
            y_true: True values of shape (n_outputs, batch_size)
            
        Returns:
            Loss values of shape (batch_size,)
        """
        return 0.5 * np.sum(np.square(y_pred - y_true), axis=0)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Backward pass of MSE loss
        
        Args:
            y_pred: Predicted values of shape (n_outputs, batch_size)
            y_true: True values of shape (n_outputs, batch_size)
            
        Returns:
            Gradient of shape (n_outputs, batch_size)
        """
        return y_pred - y_true