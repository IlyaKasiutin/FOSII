from ..module import Module
import numpy as np

class ReLU(Module): 
    def __init__(self):
        super().__init__()
        self.output = None
    
    def forward(self, x: np.ndarray):
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, output_grad: np.ndarray):
        return output_grad * (self.output > 0)
    