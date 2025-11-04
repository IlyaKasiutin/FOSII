from ..module import Module
from ..utils import he_initialization, init_biases
from ..layers.tanh import Tanh
from ..layers.softmax import Softmax
import numpy as np


class MultiplyGate:
    def forward(self, W: np.ndarray, x: np.ndarray):
        return W @ x


    def backward(self, W: np.ndarray, x: np.ndarray, output_grad: np.ndarray):
        # dW = output_grad.T @ x
        # dx = W.T @ output_grad
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(output_grad)), np.asmatrix(x)))
        dx = np.dot(np.transpose(W), output_grad)
        return dW, dx


class AddGate:
    def forward(self, x1: np.ndarray, x2: np.ndarray):
        return x1 + x2


    def backward(self, x1, x2, output_grad: np.ndarray):
        dx1 = output_grad * np.ones_like(x1)
        dx2 = output_grad * np.ones_like(x2)
        return dx1, dx2


class RNN(Module):
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int):
        """https://github.com/gy910210/rnn-from-scratch?tab=readme-ov-file"""
        super().__init__()
        self.U = he_initialization(n_inputs, n_hidden)  # x
        self.W = he_initialization(n_hidden, n_hidden)  # s
        self.V = he_initialization(n_hidden, n_outputs)  # y
        self._params = {"weights": [self.U, self.W, self.V]}
        self.U_grad = np.zeros_like(self.U)
        self.W_grad = np.zeros_like(self.W)
        self.V_grad = np.zeros_like(self.V)
        self.multiply_gate = MultiplyGate()
        self.add_gate = AddGate()
        self.tanh = Tanh()
        self.softmax = Softmax()
        self.prev_s = np.zeros(n_hidden)
        self.output = None
        self.input = None

    def single_step(self, x: np.ndarray) -> np.ndarray:
        self.input = x # (n_in, batch)
        # print(f"{x.shape=} {self.prev_s.shape=} {self.U.shape=} {self.W.shape=} {self.V.shape=}")
        self.mulu = self.multiply_gate.forward(self.U, x)
        self.mulw = self.multiply_gate.forward(self.W, self.prev_s)
        self.add = self.add_gate.forward(self.mulw, self.mulu)
        self.s = self.tanh.forward(self.add)
        self.mulv = self.multiply_gate.forward(self.V, self.s)

        return self.mulv
    

    def backward(self, x: np.ndarray, prev_s: np.ndarray, diff_s: np.ndarray, dmulv: np.ndarray):
        # print(f"{self.V.shape=} {self.s.shape=} {dmulv.shape=}")
        dV, dsv = self.multiply_gate.backward(self.V, self.s, dmulv)
        ds = dsv + diff_s
        # dadd = self.tanh.backward(self.add, ds)
        # print(f"{self.add.shape=} {ds.shape=}")
        dadd = self.tanh.backward(ds)
        # print(f"{self.mulw.shape=} {self.mulu.shape=} {dadd.shape=}")
        dmulw, dmulu = self.add_gate.backward(self.mulw, self.mulu, dadd)
        # print(f"{self.W.shape=} {prev_s.shape=} {dmulw.shape=}")
        dW, dprev_s = self.multiply_gate.backward(self.W, prev_s, dmulw)
        # print(f"{self.U.shape=} {x.shape=} {dmulu.shape=}")
        dU, dx = self.multiply_gate.backward(self.U, x, dmulu)
        return (dprev_s, dU, dW, dV)
    
    def update_params(self, learning_rate: float = 0.001):
        self.W -= learning_rate * self.W_grad
        self.bias -= learning_rate * self.bias_grad
        self.U -= learning_rate * self.U_grad
        self.V -= learning_rate * self.V_grad
