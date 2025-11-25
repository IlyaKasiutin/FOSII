import numpy as np
from typing import Optional
from ..module import Module
from ..layers.linear import Linear
from ..layers.relu import ReLU
from ..layers.layer_norm import LayerNorm
from ..utils import he_initialization, init_biases   



class MultiHeadAttention(Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        # Query, Key, Value projections
        self.W_q = he_initialization(d_model, d_model)
        self.W_k = he_initialization(d_model, d_model)
        self.W_v = he_initialization(d_model, d_model)
        self.W_o = he_initialization(d_model, d_model)
        
        self._params = {
            "W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o
        }
        
        # Gradients
        self.W_q_grad = np.zeros_like(self.W_q)
        self.W_k_grad = np.zeros_like(self.W_k)
        self.W_v_grad = np.zeros_like(self.W_v)
        self.W_o_grad = np.zeros_like(self.W_o)
        
        # Cache for backward
        self.input = None
        self.q = None
        self.k = None
        self.v = None
        self.attn_scores = None
        self.attn_weights = None
        self.attn_output = None
        self.output = None
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input of shape (d_model, seq_len)
            mask: Optional attention mask of shape (seq_len, seq_len)
        
        Returns:
            Output of shape (d_model, seq_len)
        """
        self.input = x
        seq_len = x.shape[1]
        batch_size = 1  # For simplicity, assuming single batch
        
        # Project to Q, K, V: (d_model, seq_len)
        self.q = self.W_q @ x  # (d_model, seq_len)
        self.k = self.W_k @ x  # (d_model, seq_len)
        self.v = self.W_v @ x  # (d_model, seq_len)
        
        # Reshape for multi-head: (n_heads, d_k, seq_len)
        q_heads = self.q.reshape(self.n_heads, self.d_k, seq_len)
        k_heads = self.k.reshape(self.n_heads, self.d_k, seq_len)
        v_heads = self.v.reshape(self.n_heads, self.d_k, seq_len)
        
        # Scaled dot-product attention: (n_heads, seq_len, seq_len)
        # Q @ K^T: (n_heads, d_k, seq_len) @ (n_heads, seq_len, d_k) = (n_heads, seq_len, seq_len)
        self.attn_scores = np.einsum('hds,hdm->hsm', q_heads, k_heads) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            self.attn_scores = np.where(mask == 0, -1e9, self.attn_scores)
        
        # Softmax over the last dimension
        # Numerical stability
        exp_scores = np.exp(self.attn_scores - np.max(self.attn_scores, axis=-1, keepdims=True))
        self.attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply attention to values: (n_heads, d_k, seq_len)
        # (n_heads, seq_len, seq_len) @ (n_heads, d_k, seq_len) -> need to transpose v_heads
        v_heads_T = np.transpose(v_heads, (0, 2, 1))  # (n_heads, seq_len, d_k)
        attn_output_heads = self.attn_weights @ v_heads_T  # (n_heads, seq_len, d_k)
        attn_output_heads = np.transpose(attn_output_heads, (0, 2, 1))  # (n_heads, d_k, seq_len)
        
        # Concatenate heads: (d_model, seq_len)
        self.attn_output = attn_output_heads.reshape(self.d_model, seq_len)
        
        # Output projection
        self.output = self.W_o @ self.attn_output
        
        return self.output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of multi-head attention.
        
        Args:
            output_grad: Gradient from next layer, shape (d_model, seq_len)
        
        Returns:
            Gradient with respect to input, shape (d_model, seq_len)
        """
        seq_len = self.input.shape[1]
        
        # Gradient through output projection
        dattn_output = self.W_o.T @ output_grad
        dW_o = output_grad @ self.attn_output.T
        self.W_o_grad = dW_o
        
        # Reshape attention output gradient: (n_heads, d_k, seq_len)
        dattn_output_heads = dattn_output.reshape(self.n_heads, self.d_k, seq_len)
        
        # Get V heads for gradient computation
        v_heads = self.v.reshape(self.n_heads, self.d_k, seq_len)
        
        # Gradient through attention weights: dattn_weights = dattn_output @ v^T
        # dattn_output_heads: (n_heads, d_k, seq_len), v_heads: (n_heads, d_k, seq_len)
        # We need: (n_heads, seq_len, seq_len) = (n_heads, seq_len, d_k) @ (n_heads, d_k, seq_len)
        dattn_output_heads_T = np.transpose(dattn_output_heads, (0, 2, 1))  # (n_heads, seq_len, d_k)
        v_heads_T = np.transpose(v_heads, (0, 2, 1))  # (n_heads, seq_len, d_k)
        dattn_weights = dattn_output_heads_T @ v_heads  # (n_heads, seq_len, d_k) @ (n_heads, d_k, seq_len) = (n_heads, seq_len, seq_len)
        
        # Gradient through softmax: Jacobian of softmax
        # For softmax: dL/dx_i = softmax_i * (dL/dy_i - sum_j(softmax_j * dL/dy_j))
        dattn_scores = self.attn_weights * (dattn_weights - np.sum(self.attn_weights * dattn_weights, axis=-1, keepdims=True))
        dattn_scores = dattn_scores / np.sqrt(self.d_k)
        
        # Gradient through Q, K projections
        q_heads = self.q.reshape(self.n_heads, self.d_k, seq_len)
        k_heads = self.k.reshape(self.n_heads, self.d_k, seq_len)
        
        # dattn_scores: (n_heads, seq_len, seq_len)
        # For Q: dq = dattn_scores @ k^T
        # (n_heads, seq_len, seq_len) @ (n_heads, seq_len, d_k) = (n_heads, seq_len, d_k)
        k_heads_T = np.transpose(k_heads, (0, 2, 1))  # (n_heads, seq_len, d_k)
        dq_heads_T = dattn_scores @ k_heads_T  # (n_heads, seq_len, seq_len) @ (n_heads, seq_len, d_k) = (n_heads, seq_len, d_k)
        dq_heads = np.transpose(dq_heads_T, (0, 2, 1))  # (n_heads, d_k, seq_len)
        
        # For K: dk = dattn_scores^T @ q^T
        # (n_heads, seq_len, seq_len) @ (n_heads, seq_len, d_k) = (n_heads, seq_len, d_k)
        q_heads_T = np.transpose(q_heads, (0, 2, 1))  # (n_heads, seq_len, d_k)
        dk_heads_T = np.transpose(dattn_scores, (0, 2, 1)) @ q_heads_T  # (n_heads, seq_len, seq_len) @ (n_heads, seq_len, d_k)
        dk_heads = np.transpose(dk_heads_T, (0, 2, 1))  # (n_heads, d_k, seq_len)
        
        dq = dq_heads.reshape(self.d_model, seq_len)
        dk = dk_heads.reshape(self.d_model, seq_len)
        
        # Gradient through V: dv = attn_weights^T @ dattn_output
        # (n_heads, seq_len, seq_len) @ (n_heads, seq_len, d_k) = (n_heads, seq_len, d_k)
        dv_heads_T = np.transpose(self.attn_weights, (0, 2, 1)) @ dattn_output_heads_T  # (n_heads, seq_len, seq_len) @ (n_heads, seq_len, d_k)
        dv_heads = np.transpose(dv_heads_T, (0, 2, 1))  # (n_heads, d_k, seq_len)
        dv = dv_heads.reshape(self.d_model, seq_len)
        
        # Gradient through input projections
        dx_q = self.W_q.T @ dq
        dx_k = self.W_k.T @ dk
        dx_v = self.W_v.T @ dv
        
        dx = dx_q + dx_k + dx_v
        
        # Parameter gradients
        self.W_q_grad = dq @ self.input.T
        self.W_k_grad = dk @ self.input.T
        self.W_v_grad = dv @ self.input.T
        
        return dx
    
    def update_params(self, learning_rate: float = 0.001):
        self.W_q -= learning_rate * self.W_q_grad
        self.W_k -= learning_rate * self.W_k_grad
        self.W_v -= learning_rate * self.W_v_grad
        self.W_o -= learning_rate * self.W_o_grad


class FeedForward(Module):
    """Feed-forward network with two linear layers and ReLU activation."""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.relu = ReLU()
        self._modules = {"linear1": self.linear1, "linear2": self.linear2, "relu": self.relu}
        self.input = None
        self.output = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input of shape (d_model, seq_len)
        
        Returns:
            Output of shape (d_model, seq_len)
        """
        self.input = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        self.output = out
        return self.output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of feed-forward network.
        
        Args:
            output_grad: Gradient from next layer, shape (d_model, seq_len)
        
        Returns:
            Gradient with respect to input, shape (d_model, seq_len)
        """
        out = self.linear2.backward(output_grad)
        out = self.relu.backward(out)
        out = self.linear1.backward(out)
        return out
    
    def update_params(self, learning_rate: float = 0.001):
        self.linear1.update_params(learning_rate)
        self.linear2.update_params(learning_rate)
    
    def zero_grad(self):
        self.linear1.zero_grad()
        self.linear2.zero_grad()


class TransformerBlock(Module):
    """Transformer encoder block with self-attention and feed-forward."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self._modules = {
            "self_attn": self.self_attn,
            "feed_forward": self.feed_forward,
            "norm1": self.norm1,
            "norm2": self.norm2
        }
        self.input = None
        self.output = None
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input of shape (d_model, seq_len)
            mask: Optional attention mask
        
        Returns:
            Output of shape (d_model, seq_len)
        """
        self.input = x
        
        # Self-attention with residual connection and layer norm
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, mask)
        x = x + attn_output  # Residual connection
        
        # Feed-forward with residual connection and layer norm
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + ff_output  # Residual connection
        
        self.output = x
        return self.output
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Backward pass of transformer block.
        
        Args:
            output_grad: Gradient from next layer, shape (d_model, seq_len)
        
        Returns:
            Gradient with respect to input, shape (d_model, seq_len)
        """
        # Backward through feed-forward with residual connection
        # The output is: x + feed_forward(norm2(x))
        # So gradient flows through both paths
        dff_path = self.feed_forward.backward(output_grad)  # Gradient through FF path
        dnorm2 = self.norm2.backward(dff_path)
        dx_after_attn = output_grad + dnorm2  # Gradient through residual: both paths
        
        # Backward through attention with residual connection
        # The output is: x + attention(norm1(x))
        # So gradient flows through both paths
        dattn_path = self.self_attn.backward(dx_after_attn)  # Gradient through attention path
        dnorm1 = self.norm1.backward(dattn_path)
        dx = dx_after_attn + dnorm1  # Gradient through residual: both paths
        
        return dx
    
    def update_params(self, learning_rate: float = 0.001):
        self.self_attn.update_params(learning_rate)
        self.feed_forward.update_params(learning_rate)
        self.norm1.update_params(learning_rate)
        self.norm2.update_params(learning_rate)
    
    def zero_grad(self):
        self.self_attn.W_q_grad.fill(0)
        self.self_attn.W_k_grad.fill(0)
        self.self_attn.W_v_grad.fill(0)
        self.self_attn.W_o_grad.fill(0)
        self.feed_forward.zero_grad()
        self.norm1.gamma_grad.fill(0)
        self.norm1.beta_grad.fill(0)
        self.norm2.gamma_grad.fill(0)
        self.norm2.beta_grad.fill(0)

