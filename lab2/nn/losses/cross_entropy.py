import numpy as np
from ..module import Module


class CrossEntropy(Module):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred_stable = np.max(y_pred, axis=0, keepdims=True)
        exp_pred = np.exp(y_pred_stable)
        softmax_pred = exp_pred / np.sum(exp_pred, axis=0, keepdims=True)

        epsilon = 1e-15
        softmax_pred = np.clip(softmax_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(softmax_pred), axis=0)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_pred_stable = np.max(y_pred, axis=0, keepdims=True)
        exp_pred = np.exp(y_pred_stable)
        softmax_pred = exp_pred / np.sum(exp_pred, axis=0, keepdims=True)
        
        return softmax_pred - y_true