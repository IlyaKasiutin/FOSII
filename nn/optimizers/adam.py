from .base import Optimizer
import numpy as np


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Combines advantages of AdaGrad and RMSProp by maintaining 
    exponential moving averages of both the gradients and their squared values.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate (alpha in Adam paper)
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # State dictionaries: maps parameter arrays to their moments
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Time step
    
    def step(self, model) -> None:
        """
        Perform a single Adam optimization step.
        
        Args:
            model: Model instance (Module) containing layers with gradients
        """
        self.t += 1
        
        for param_name, param, grad in self._get_param_grad_pairs(model):
            # print(f"{param_name=} {param=} {grad=}")
            # Use id() of parameter array as key to uniquely identify it
            param_id = id(param)
            
            # Initialize moment estimates if this is the first time we see this parameter
            if param_id not in self.m:
                self.m[param_id] = np.zeros_like(param)
                self.v[param_id] = np.zeros_like(param)
            
            # Update biased first moment estimate
            self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def state_dict(self):
        """
        Returns the state of the optimizer.
        
        Returns:
            Dictionary containing optimizer state
        """
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'm': self.m,
            'v': self.v,
            't': self.t
        }
    
    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        
        Args:
            state_dict: Dictionary containing optimizer state
        """
        self.learning_rate = state_dict['learning_rate']
        self.beta1 = state_dict['beta1']
        self.beta2 = state_dict['beta2']
        self.epsilon = state_dict['epsilon']
        self.m = state_dict['m']
        self.v = state_dict['v']
        self.t = state_dict['t']

