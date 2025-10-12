import numpy as np
from typing import Callable
from tqdm import tqdm
rng = np.random.default_rng(51)


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def sigmoid_diff(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def mse(y_pred: float, y_true: float) -> float:
    err = 1 / 2 * np.square(y_pred - y_true)
    return err


def mse_diff(y_pred: float, y_true: float) -> float:
    return y_pred - y_true


class MLP:
    def __init__(self, dims: list[int], activation_info: dict[str, Callable], loss_info: dict):
        self.dims = dims
        self.activation = np.vectorize(activation_info["activation"])
        self.activation_diff = np.vectorize(activation_info["activation_diff"])
        self.loss = np.vectorize(loss_info["loss"])
        self.loss_diff = np.vectorize(loss_info["loss_diff"])
        self.weights = [np.random.randn(dims[i + 1], dims[i]) for i in range(len(dims) - 1)]
        self.biases = [np.random.randn(dims[i], 1) for i in range(1, len(dims))]
        self.hidden_states = [np.zeros_like(bias_layer) for bias_layer in self.biases]  # состояния нейронов (для обновления весов)
        self.z_states = [np.zeros_like(bias_layer) for bias_layer in self.biases]  # состояния z в нейроне до функции активации (для обновления весов)
        self.weights_grads = [np.zeros_like(weights_layer) for weights_layer in self.weights]  # градиенты весов
        self.bias_grads = [np.zeros_like(bias_layer) for bias_layer in self.biases ]  # градиенты смещений
        self.inputs = None


    def _forward(self, X: np.ndarray, compute_grad: bool = True) -> np.ndarray:
        """
        Result: np.ndarray((n_samples, n_outputs))
        """

        if X.ndim == 1:  # подан один объект
            X = X.reshape(1, -1)
        
        result = X.T

        if compute_grad:
            self.inputs = result

        for layer_idx, (weights, biases) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            result = np.dot(weights, result) + biases

            if compute_grad:
                self.z_states[layer_idx] = result
                self.hidden_states[layer_idx] = self.activation(result)

            result = self.activation(result)

        transposed_output = np.dot(self.weights[-1], result) + self.biases[-1]
        self.hidden_states[-1] = transposed_output
        self.z_states[-1] = self.hidden_states[-1]

        return transposed_output.T


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


    def _backprop(self, y_true: np.ndarray):
        """
        Inputs: y_true: np.ndarray(n_samples)
        """

        y_true_extended = np.zeros((len(y_true), self.dims[-1]))
        y_true_extended[np.arange(len(y_true_extended)), y_true] = 1

        prev_delta = None
        for i in range(len(self.dims) - 2, -1, -1):
            if i == len(self.dims) - 2:
                delta = self.loss_diff(self.hidden_states[i], y_true_extended.T)  # (n_outputs, n_samples)
                self.weights_grads[i] = np.dot(delta, self.hidden_states[i - 1].T) / delta.shape[-1]  # (n_outputs, n_inputs)
                self.bias_grads[i] = np.sum(delta, axis=1, keepdims=True) / delta.shape[-1]  # (n_ouputs, 1)

                prev_delta = np.sum(delta, axis=1, keepdims=True) / delta.shape[-1] # (n_outputs, n_samples)
            elif i > 0:
                delta = np.dot(self.weights[i + 1].T, prev_delta)  # (n_outputs, 1) с предыдущего слоя 
                sigmoid_diff = self.activation_diff(self.z_states[i])  # (n_outputs, n_samples) теперь берем с текущего слоя
                sigmoid_diff = np.sum(sigmoid_diff, axis=1, keepdims=True)  # (n_outputs, 1)
                self.weights_grads[i] = np.dot(delta * sigmoid_diff,  np.sum(self.hidden_states[i - 1], axis=1, keepdims=True).T)  # (n_outputs, n_inputs)
                self.bias_grads[i] = delta * sigmoid_diff

                prev_delta = delta * sigmoid_diff
            else:
                delta = np.dot(self.weights[i + 1].T, prev_delta)  # (n_outputs, 1) с предыдущего слоя                 
                sigmoid_diff = self.activation_diff(self.z_states[i])  # (n_outputs, n_samples) теперь берем с текущего слоя
                sigmoid_diff = np.sum(sigmoid_diff, axis=1, keepdims=True)  # (n_outputs, 1)              
                self.weights_grads[i] = np.dot(delta * sigmoid_diff,  np.sum(self.inputs, axis=1, keepdims=True).T)  # (n_outputs, n_inputs)
                self.bias_grads[i] = delta * sigmoid_diff

                prev_delta = delta * sigmoid_diff


    def _init_weights(self):
        net_in = self.dims[0]
        net_out = self.dims[-1]
        limit = np.sqrt(6. / (net_in + net_out))
        return [rng.uniform(-limit, limit + 1e-5, size=(self.dims[i+1], self.dims[i])) for i in range(len(self.dims) - 1)]


    def _init_biases(self):
        return [rng.random((self.dims[i], 1)) * 2 - 1 for i in range(1, len(self.dims))]
                

    def _update_weights(self, learning_rate: float = 1e-3):
        for i in range(len(self.dims) - 2, -1, -1):
            self.weights[i] -= learning_rate * self.weights_grads[i]
            self.biases[i] -= learning_rate * self.bias_grads[i]


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
