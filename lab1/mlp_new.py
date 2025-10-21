import numpy as np
from typing import Callable
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath("")))

from nn.layers.sigmoid import Sigmoid
from nn.layers.linear import Linear
from nn.losses.mse import MSE
from nn.module import Module

rng = np.random.default_rng(51)


class MLP:
    def __init__(self):
        self.layers: list[Module] = [
            Linear(21, 10),
            Sigmoid(),
            Linear(10, 18),
            Sigmoid(),
            Linear(18, 2)
        ]


    def _forward(self, X: np.ndarray, compute_grad: bool = True) -> np.ndarray:
        """
        Result: np.ndarray((n_samples, n_outputs))
        """

        if X.ndim == 1:  # подан один объект
            X = X.reshape(1, -1)
        
        result = X.T

        for layer in self.layers:
            result = layer(result)

        return result.T


    def _backward(self, loss_grad: np.ndarray) -> None:
        output_grad = loss_grad
        for layer in self.layers[::-1]:
            output_grad = layer.backward(output_grad)

    def _update_weights(self, lr: float = 0.001) -> None:
        for layer in self.layers:
            layer.


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Result: np.ndarray((n_samples, n_outputs))
        """

        return self._forward(X, compute_grad=False)
    

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_true_extended = np.zeros((len(y_true), self.dims[-1]))
        y_true_extended[np.arange(len(y_true_extended)), y_true] = 1

        losses = self.loss(y_pred, y_true_extended)
        return losses.mean()


    def _zero_grad(self):
        for i in range(len(self.weights_grads)):
            self.weights_grads[i] = np.zeros_like(self.weights_grads[i])
            self.bias_grads[i] = np.zeros_like(self.bias_grads[i])

    @staticmethod
    def _shuffle_samples(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        new_indexes = np.random.permutation(len(y))
        return X[new_indexes], y[new_indexes]

    @staticmethod
    def _get_random_idx(max_idx: int, n: int) -> np.ndarray:
        if n < max_idx:
            return np.random.choice(max_idx, n, replace=False)
        return np.arange(max_idx)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iters=10,
        batch_size=1000,
        stop_threshold: float = 0.1,
        learning_rate: float = 1e-3,
    ) -> list[float]:
        losses = []
        self.weights = self._init_weights()
        self.biases = self._init_biases()

        for iter in tqdm(range(n_iters), total=n_iters):
            current_weights = sum(np.sum(weights**2) for weights in self.weights)
            loss = self.compute_loss(self.predict(X_train), y_train)

            batch_idx = self._shuffle_samples(X_train, y_train)
            untrained_X, untrained_y = X_train, y_train

            while len(untrained_X):
                batch_idx = self._get_random_idx(len(untrained_X), batch_size)

                batch_X, batch_y = untrained_X[batch_idx], untrained_y[batch_idx]
                untrained_X = np.delete(untrained_X, batch_idx, axis=0)
                untrained_y = np.delete(untrained_y, batch_idx)

                self._forward(batch_X, compute_grad=True)
                self._backprop(batch_y)
                self._update_weights(learning_rate)
                self._zero_grad()

            new_weights = sum(np.sum(weights**2) for weights in self.weights)
            if np.abs(new_weights - current_weights) / new_weights < stop_threshold:
                print("Early stopping by weights")
                break

            new_loss = self.compute_loss(self.predict(X_train), y_train)
            if np.abs(new_loss - loss) / loss < stop_threshold:
                print("Early stopping by loss")
                break
            losses.append(loss)

        print(f"Final loss: {loss=}")
        return losses
