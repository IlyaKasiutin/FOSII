from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """
    Base class for all optimizers.
    """
    
    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.learning_rate = learning_rate
    
    @abstractmethod
    def step(self, model) -> None:
        """
        Perform a single optimization step.
        
        Args:
            model: Model instance (Module) containing layers with gradients
        """
        pass
    
    def _get_param_grad_pairs(self, model):
        """
        Helper method to extract (parameter, gradient) pairs from a model.
        
        Yields:
            tuples of (param_name, param_array, grad_array) for each parameter
        """
        # Iterate through all layers in the model
        layers = []
        
        # Collect all layers from the model
        # Priority: all_layers > (conv_layers + fc_layers) > layers > _modules
        if hasattr(model, 'all_layers'):
            layers.extend(model.all_layers)
        else:
            if hasattr(model, 'conv_layers'):
                layers.extend(model.conv_layers)
            if hasattr(model, 'fc_layers'):
                layers.extend(model.fc_layers)
            if hasattr(model, 'layers'):
                layers.extend(model.layers)
        
        # Fallback: try to get from _modules (always check as fallback)
        if not layers and hasattr(model, '_modules'):
            for module in model._modules.values():
                if (hasattr(module, 'W_grad') or hasattr(module, 'bias_grad') or 
                    hasattr(module, 'U_grad') or hasattr(module, 'V_grad')):
                    layers.append(module)
        
        # Yield parameter-gradient pairs
        # Only yield for layers that actually have parameters (Linear, Conv2d, RNN, etc.)
        for layer in layers:
            if hasattr(layer, 'W') and hasattr(layer, 'W_grad'):
                yield ('W', layer.W, layer.W_grad)
            if hasattr(layer, 'bias') and hasattr(layer, 'bias_grad'):
                yield ('bias', layer.bias, layer.bias_grad)
            # RNN-specific parameters
            if hasattr(layer, 'U') and hasattr(layer, 'U_grad'):
                yield ('U', layer.U, layer.U_grad)
            if hasattr(layer, 'V') and hasattr(layer, 'V_grad'):
                yield ('V', layer.V, layer.V_grad)

