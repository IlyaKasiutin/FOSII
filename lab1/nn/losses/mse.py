import numpy as np
from nn.module import Module
from typing import Union


def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_true_extended = np.zeros((len(y_true), self.dims[-1]))
        y_true_extended[np.arange(len(y_true_extended)), y_true] = 1

        losses = self.loss(y_pred, y_true_extended)
        return losses.mean()


def mse(y_pred: float, y_true: float) -> float:
    err = 1 / 2 * np.square(y_pred - y_true)
    return err


def mse_diff(y_pred: float, y_true: float) -> float:
    return y_pred - y_true


class MSE(Module):
    def forward(self, y_pred: np.ndarray | float, y_true: np.ndarray | float) -> float | np.ndarray:
        return 1 / 2 * np.square(y_pred - y_true)

    def backward(self, y_pred: float, y_true: float) -> float:
        return y_pred - y_true   