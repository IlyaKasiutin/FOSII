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
    """Element-wise multiplication gate for LSTM gates"""
    def forward(self, x1: np.ndarray, x2: np.ndarray):
        return x1 * x2

    def backward(self, x1: np.ndarray, x2: np.ndarray, output_grad: np.ndarray):
        dx1 = output_grad * x2
        dx2 = output_grad * x1
        return dx1, dx2


class LSTM(Module):
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int):
        """
        LSTM (Long Short-Term Memory) block.
        
        Architecture:
        - Forget gate: f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)
        - Input gate: i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)
        - Candidate cell state: c_tilde = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
        - Cell state: c_t = f_t * c_{t-1} + i_t * c_tilde
        - Output gate: o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)
        - Hidden state: h_t = o_t * tanh(c_t)
        - Output: y_t = V @ h_t
        https://en.wikipedia.org/wiki/Long_short-term_memory
        """
        super().__init__()
        
        # Forget gate weights
        self.W_f = he_initialization(n_inputs, n_hidden)  # input to forget gate
        self.U_f = he_initialization(n_hidden, n_hidden)   # hidden to forget gate
        self.b_f = init_biases(n_hidden).flatten()  # bias for forget gate
        
        # Input gate weights
        self.W_i = he_initialization(n_inputs, n_hidden)  # input to input gate
        self.U_i = he_initialization(n_hidden, n_hidden)  # hidden to input gate
        self.b_i = init_biases(n_hidden).flatten()  # bias for input gate
        
        # Candidate cell state weights
        self.W_c = he_initialization(n_inputs, n_hidden)  # input to candidate
        self.U_c = he_initialization(n_hidden, n_hidden)  # hidden to candidate
        self.b_c = init_biases(n_hidden).flatten()  # bias for candidate
        
        # Output gate weights
        self.W_o = he_initialization(n_inputs, n_hidden)  # input to output gate
        self.U_o = he_initialization(n_hidden, n_hidden)  # hidden to output gate
        self.b_o = init_biases(n_hidden).flatten()  # bias for output gate
        
        # Output weights
        self.V = he_initialization(n_hidden, n_outputs)    # output from hidden
        
        self._params = {"weights": [self.W_f, self.U_f, self.b_f, self.W_i, self.U_i, self.b_i,
                                   self.W_c, self.U_c, self.b_c, self.W_o, self.U_o, self.b_o, self.V]}
        
        # Gradients
        self.W_f_grad = np.zeros_like(self.W_f)
        self.U_f_grad = np.zeros_like(self.U_f)
        self.b_f_grad = np.zeros_like(self.b_f)
        
        self.W_i_grad = np.zeros_like(self.W_i)
        self.U_i_grad = np.zeros_like(self.U_i)
        self.b_i_grad = np.zeros_like(self.b_i)
        
        self.W_c_grad = np.zeros_like(self.W_c)
        self.U_c_grad = np.zeros_like(self.U_c)
        self.b_c_grad = np.zeros_like(self.b_c)
        
        self.W_o_grad = np.zeros_like(self.W_o)
        self.U_o_grad = np.zeros_like(self.U_o)
        self.b_o_grad = np.zeros_like(self.b_o)
        
        self.V_grad = np.zeros_like(self.V)
        
        # Gates and activations
        self.multiply_gate = MultiplyGate()
        self.add_gate = AddGate()
        self.hadamard_gate = HadamardGate()
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        
        # State
        self.prev_h = np.zeros(n_hidden)
        self.prev_c = np.zeros(n_hidden)
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
        
        # Forget gate: f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)
        self.Wf_x = self.multiply_gate.forward(self.W_f, x)
        self.Uf_h = self.multiply_gate.forward(self.U_f, self.prev_h)
        forget_sum = self.add_gate.forward(self.Wf_x, self.Uf_h)
        forget_sum = self.add_gate.forward(forget_sum, self.b_f)
        self.f_t = self.sigmoid.forward(forget_sum)
        
        # Input gate: i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)
        self.Wi_x = self.multiply_gate.forward(self.W_i, x)
        self.Ui_h = self.multiply_gate.forward(self.U_i, self.prev_h)
        input_sum = self.add_gate.forward(self.Wi_x, self.Ui_h)
        input_sum = self.add_gate.forward(input_sum, self.b_i)
        self.i_t = self.sigmoid.forward(input_sum)
        
        # Candidate cell state: c_tilde = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
        self.Wc_x = self.multiply_gate.forward(self.W_c, x)
        self.Uc_h = self.multiply_gate.forward(self.U_c, self.prev_h)
        candidate_sum = self.add_gate.forward(self.Wc_x, self.Uc_h)
        candidate_sum = self.add_gate.forward(candidate_sum, self.b_c)
        self.c_tilde = self.tanh.forward(candidate_sum)
        
        # Cell state: c_t = f_t * c_{t-1} + i_t * c_tilde
        self.f_c = self.hadamard_gate.forward(self.f_t, self.prev_c)
        self.i_c_tilde = self.hadamard_gate.forward(self.i_t, self.c_tilde)
        self.c_t = self.add_gate.forward(self.f_c, self.i_c_tilde)
        
        # Output gate: o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)
        self.Wo_x = self.multiply_gate.forward(self.W_o, x)
        self.Uo_h = self.multiply_gate.forward(self.U_o, self.prev_h)
        output_sum = self.add_gate.forward(self.Wo_x, self.Uo_h)
        output_sum = self.add_gate.forward(output_sum, self.b_o)
        self.o_t = self.sigmoid.forward(output_sum)
        
        # Hidden state: h_t = o_t * tanh(c_t)
        self.tanh_c = self.tanh.forward(self.c_t)
        self.h_t = self.hadamard_gate.forward(self.o_t, self.tanh_c)
        
        # Output: y_t = V @ h_t
        self.mulv = self.multiply_gate.forward(self.V, self.h_t)
        
        # Update previous states for next step
        self.prev_h = self.h_t.copy()
        self.prev_c = self.c_t.copy()
        
        return self.mulv

    def backward(self, x: np.ndarray, prev_h: np.ndarray, prev_c: np.ndarray, 
                 diff_h: np.ndarray, diff_c: np.ndarray, dmulv: np.ndarray):
        """
        Backward pass for a single time step.
        
        Args:
            x: Input at time t
            prev_h: Previous hidden state (before this step)
            prev_c: Previous cell state (before this step)
            diff_h: Gradient flowing into hidden state from future time steps
            diff_c: Gradient flowing into cell state from future time steps
            dmulv: Gradient from output
        
        Returns:
            dprev_h: Gradient w.r.t. previous hidden state
            dprev_c: Gradient w.r.t. previous cell state
            dW_f, dU_f, db_f, dW_i, dU_i, db_i, dW_c, dU_c, db_c, dW_o, dU_o, db_o, dV: Parameter gradients
        """
        # Output layer: dmulv -> h_t (use cached h_t from forward)
        dV, dh_v = self.multiply_gate.backward(self.V, self.h_t, dmulv)
        dh = dh_v + diff_h
        
        # Hidden state: h_t = o_t * tanh(c_t)
        # Use cached values: self.o_t, self.tanh_c
        do_t, dtanh_c = self.hadamard_gate.backward(self.o_t, self.tanh_c, dh)
        
        # Output gate: o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)
        # Use cached o_t
        d_output_sum = self.sigmoid.backward(do_t)
        dWo_x, dUo_h = self.add_gate.backward(self.Wo_x, self.Uo_h, d_output_sum)
        dWo_x, db_o = self.add_gate.backward(dWo_x, self.b_o, d_output_sum)
        dW_o, dx_o = self.multiply_gate.backward(self.W_o, x, dWo_x)
        dU_o, dprev_h_o = self.multiply_gate.backward(self.U_o, prev_h, dUo_h)
        
        # tanh(c_t) - use cached tanh_c
        dc_t_tanh = self.tanh.backward(dtanh_c)
        # Combine with gradient from next time step
        dc_t = dc_t_tanh + diff_c
        
        # Cell state: c_t = f_t * c_{t-1} + i_t * c_tilde
        # Use cached values: self.f_c, self.i_c_tilde
        df_c, di_c_tilde = self.add_gate.backward(self.f_c, self.i_c_tilde, dc_t)
        
        # f_t * c_{t-1}
        df_t, dprev_c_f = self.hadamard_gate.backward(self.f_t, prev_c, df_c)
        
        # i_t * c_tilde
        di_t, dc_tilde = self.hadamard_gate.backward(self.i_t, self.c_tilde, di_c_tilde)
        
        # Forget gate: f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)
        # Use cached f_t
        d_forget_sum = self.sigmoid.backward(df_t)
        dWf_x, dUf_h = self.add_gate.backward(self.Wf_x, self.Uf_h, d_forget_sum)
        dWf_x, db_f = self.add_gate.backward(dWf_x, self.b_f, d_forget_sum)
        dW_f, dx_f = self.multiply_gate.backward(self.W_f, x, dWf_x)
        dU_f, dprev_h_f = self.multiply_gate.backward(self.U_f, prev_h, dUf_h)
        
        # Input gate: i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)
        # Use cached i_t
        d_input_sum = self.sigmoid.backward(di_t)
        dWi_x, dUi_h = self.add_gate.backward(self.Wi_x, self.Ui_h, d_input_sum)
        dWi_x, db_i = self.add_gate.backward(dWi_x, self.b_i, d_input_sum)
        dW_i, dx_i = self.multiply_gate.backward(self.W_i, x, dWi_x)
        dU_i, dprev_h_i = self.multiply_gate.backward(self.U_i, prev_h, dUi_h)
        
        # Candidate cell state: c_tilde = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
        # Use cached c_tilde
        d_candidate_sum = self.tanh.backward(dc_tilde)
        dWc_x, dUc_h = self.add_gate.backward(self.Wc_x, self.Uc_h, d_candidate_sum)
        dWc_x, db_c = self.add_gate.backward(dWc_x, self.b_c, d_candidate_sum)
        dW_c, dx_c = self.multiply_gate.backward(self.W_c, x, dWc_x)
        dU_c, dprev_h_c = self.multiply_gate.backward(self.U_c, prev_h, dUc_h)
        
        # Combine all gradients w.r.t. input x
        dx = dx_o + dx_f + dx_i + dx_c
        
        # Combine all gradients w.r.t. prev_h
        dprev_h = dprev_h_o + dprev_h_f + dprev_h_i + dprev_h_c
        
        return (dprev_h, dprev_c_f, dW_f, dU_f, db_f, dW_i, dU_i, db_i, 
                dW_c, dU_c, db_c, dW_o, dU_o, db_o, dV)

    def update_params(self, learning_rate: float = 0.001):
        self.W_f -= learning_rate * self.W_f_grad
        self.U_f -= learning_rate * self.U_f_grad
        self.b_f -= learning_rate * self.b_f_grad
        
        self.W_i -= learning_rate * self.W_i_grad
        self.U_i -= learning_rate * self.U_i_grad
        self.b_i -= learning_rate * self.b_i_grad
        
        self.W_c -= learning_rate * self.W_c_grad
        self.U_c -= learning_rate * self.U_c_grad
        self.b_c -= learning_rate * self.b_c_grad
        
        self.W_o -= learning_rate * self.W_o_grad
        self.U_o -= learning_rate * self.U_o_grad
        self.b_o -= learning_rate * self.b_o_grad
        
        self.V -= learning_rate * self.V_grad