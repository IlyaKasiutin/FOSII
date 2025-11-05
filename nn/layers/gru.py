from ..module import Module
from ..utils import he_initialization, init_biases
from ..layers.tanh import Tanh
from ..layers.sigmoid import Sigmoid
import numpy as np


class MultiplyGate:
    def forward(self, W: np.ndarray, x: np.ndarray):
        return W @ x

    def backward(self, W: np.ndarray, x: np.ndarray, output_grad: np.ndarray):
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


class HadamardGate:
    """Element-wise multiplication gate for GRU gates"""
    def forward(self, x1: np.ndarray, x2: np.ndarray):
        return x1 * x2

    def backward(self, x1: np.ndarray, x2: np.ndarray, output_grad: np.ndarray):
        dx1 = output_grad * x2
        dx2 = output_grad * x1
        return dx1, dx2


class GRU(Module):
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int):
        """
        GRU (Gated Recurrent Unit) block.
        
        Architecture:
        - Reset gate: r_t = sigmoid(W_r @ x_t + U_r @ h_{t-1})
        - Update gate: z_t = sigmoid(W_z @ x_t + U_z @ h_{t-1})
        - Candidate: h_tilde = tanh(W_h @ x_t + U_h @ (r_t * h_{t-1}))
        - Hidden: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
        - Output: V @ h_t
        """
        super().__init__()
        
        # Reset gate weights
        self.W_r = he_initialization(n_inputs, n_hidden)  # input to reset gate
        self.U_r = he_initialization(n_hidden, n_hidden)   # hidden to reset gate
        
        # Update gate weights
        self.W_z = he_initialization(n_inputs, n_hidden)  # input to update gate
        self.U_z = he_initialization(n_hidden, n_hidden)  # hidden to update gate
        
        # Candidate hidden state weights
        self.W_h = he_initialization(n_inputs, n_hidden)  # input to candidate
        self.U_h = he_initialization(n_hidden, n_hidden)  # hidden to candidate
        
        # Output weights
        self.V = he_initialization(n_hidden, n_outputs)    # hidden to output
        
        self._params = {"weights": [self.W_r, self.U_r, self.W_z, self.U_z, 
                                   self.W_h, self.U_h, self.V]}
        
        # Gradients
        self.W_r_grad = np.zeros_like(self.W_r)
        self.U_r_grad = np.zeros_like(self.U_r)
        self.W_z_grad = np.zeros_like(self.W_z)
        self.U_z_grad = np.zeros_like(self.U_z)
        self.W_h_grad = np.zeros_like(self.W_h)
        self.U_h_grad = np.zeros_like(self.U_h)
        self.V_grad = np.zeros_like(self.V)
        
        # Gates and activations
        self.multiply_gate = MultiplyGate()
        self.add_gate = AddGate()
        self.hadamard_gate = HadamardGate()
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        
        # State
        self.prev_h = np.zeros(n_hidden)
        self.output = None
        self.input = None

    def single_step(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for a single time step.
        
        Args:
            x: Input of shape (n_inputs,) or (n_inputs, batch)
        
        Returns:
            Output of shape (n_outputs,) or (n_outputs, batch)
        """
        self.input = x
        
        # Reset gate: r_t = sigmoid(W_r @ x_t + U_r @ h_{t-1})
        self.Wr_x = self.multiply_gate.forward(self.W_r, x)
        self.Ur_h = self.multiply_gate.forward(self.U_r, self.prev_h)
        reset_sum = self.add_gate.forward(self.Wr_x, self.Ur_h)
        self.r_t = self.sigmoid.forward(reset_sum)
        
        # Update gate: z_t = sigmoid(W_z @ x_t + U_z @ h_{t-1})
        self.Wz_x = self.multiply_gate.forward(self.W_z, x)
        self.Uz_h = self.multiply_gate.forward(self.U_z, self.prev_h)
        update_sum = self.add_gate.forward(self.Wz_x, self.Uz_h)
        self.z_t = self.sigmoid.forward(update_sum)
        
        # Candidate hidden state: h_tilde = tanh(W_h @ x_t + U_h @ (r_t * h_{t-1}))
        self.r_h = self.hadamard_gate.forward(self.r_t, self.prev_h)
        self.Wh_x = self.multiply_gate.forward(self.W_h, x)
        self.Uh_r_h = self.multiply_gate.forward(self.U_h, self.r_h)
        candidate_sum = self.add_gate.forward(self.Wh_x, self.Uh_r_h)
        self.h_tilde = self.tanh.forward(candidate_sum)
        
        # Hidden state: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
        one_minus_z = 1 - self.z_t
        term1 = self.hadamard_gate.forward(one_minus_z, self.prev_h)
        term2 = self.hadamard_gate.forward(self.z_t, self.h_tilde)
        self.h_t = self.add_gate.forward(term1, term2)
        
        # Output: V @ h_t
        self.mulv = self.multiply_gate.forward(self.V, self.h_t)
        
        # Update previous hidden state for next step
        self.prev_h = self.h_t.copy()
        
        return self.mulv

    def backward(self, x: np.ndarray, prev_h: np.ndarray, diff_h: np.ndarray, dmulv: np.ndarray):
        """
        Backward pass for a single time step.
        
        Args:
            x: Input at time t
            prev_h: Previous hidden state (before this step)
            diff_h: Gradient flowing into hidden state from future time steps
            dmulv: Gradient from output
        
        Returns:
            dprev_h: Gradient w.r.t. previous hidden state
            dW_r, dU_r, dW_z, dU_z, dW_h, dU_h, dV: Parameter gradients
        """
        # Output layer: dmulv -> h_t (use cached h_t from forward)
        dV, dh_v = self.multiply_gate.backward(self.V, self.h_t, dmulv)
        dh = dh_v + diff_h
        
        # Hidden state update: h_t = (1 - z_t) * prev_h + z_t * h_tilde
        # Backprop through: term1 = (1 - z_t) * prev_h, term2 = z_t * h_tilde
        # Use cached values: self.z_t, self.h_tilde
        dterm1, dterm2 = self.add_gate.backward(1 - self.z_t, self.z_t, dh)
        
        # Term 1: (1 - z_t) * prev_h
        d_one_minus_z, dprev_h_term1 = self.hadamard_gate.backward(1 - self.z_t, prev_h, dterm1)
        d_z_term1 = -d_one_minus_z  # derivative of (1 - z_t) w.r.t. z_t
        
        # Term 2: z_t * h_tilde
        d_z_term2, dh_tilde = self.hadamard_gate.backward(self.z_t, self.h_tilde, dterm2)
        
        # Combine z_t gradients
        dz = d_z_term1 + d_z_term2
        
        # Candidate hidden state: h_tilde = tanh(W_h @ x + U_h @ (r_t * prev_h))
        # Use cached h_tilde
        d_candidate_sum = self.tanh.backward(dh_tilde)
        
        # Split gradient through addition using cached values
        dWh_x, dUh_r_h = self.add_gate.backward(self.Wh_x, self.Uh_r_h, d_candidate_sum)
        
        # U_h @ (r_t * prev_h) - use cached r_h
        dU_h, d_r_h = self.multiply_gate.backward(self.U_h, self.r_h, dUh_r_h)
        dr_t_h, dprev_h_r = self.hadamard_gate.backward(self.r_t, prev_h, d_r_h)
        
        # W_h @ x - use cached Wh_x
        dW_h, dx_h = self.multiply_gate.backward(self.W_h, x, dWh_x)
        
        # Update gate: z_t = sigmoid(W_z @ x + U_z @ prev_h)
        # Use cached z_t and cached intermediate values
        d_update_sum = self.sigmoid.backward(dz)
        dWz_x, dUz_h = self.add_gate.backward(self.Wz_x, self.Uz_h, d_update_sum)
        dW_z, dx_z = self.multiply_gate.backward(self.W_z, x, dWz_x)
        dU_z, dprev_h_z = self.multiply_gate.backward(self.U_z, prev_h, dUz_h)
        
        # Reset gate: r_t = sigmoid(W_r @ x + U_r @ prev_h)
        # Use cached r_t and cached intermediate values
        d_reset_sum = self.sigmoid.backward(dr_t_h)
        dWr_x, dUr_h = self.add_gate.backward(self.Wr_x, self.Ur_h, d_reset_sum)
        dW_r, dx_r = self.multiply_gate.backward(self.W_r, x, dWr_x)
        dU_r, dprev_h_r_gate = self.multiply_gate.backward(self.U_r, prev_h, dUr_h)
        
        # Combine all gradients w.r.t. prev_h
        dprev_h = dprev_h_term1 + dprev_h_r + dprev_h_z + dprev_h_r_gate
        
        return (dprev_h, dW_r, dU_r, dW_z, dU_z, dW_h, dU_h, dV)

    def update_params(self, learning_rate: float = 0.001):
        self.W_r -= learning_rate * self.W_r_grad
        self.U_r -= learning_rate * self.U_r_grad
        self.W_z -= learning_rate * self.W_z_grad
        self.U_z -= learning_rate * self.U_z_grad
        self.W_h -= learning_rate * self.W_h_grad
        self.U_h -= learning_rate * self.U_h_grad
        self.V -= learning_rate * self.V_grad

