from .base import Optimizer
import numpy as np


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Updates parameters using: param = param - lr * grad
    """
    
    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        super().__init__(learning_rate)
    
    def step(self, model) -> None:
        """
        Perform a single SGD optimization step.
        Updates all parameters in the model.
        
        Args:
            model: Model instance (Module) containing layers with gradients
        """
        for param_name, param, grad in self._get_param_grad_pairs(model):
            param -= self.learning_rate * grad

