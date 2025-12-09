import numpy as np


rng = np.random.default_rng(51)


def he_initialization(n_inputs: int, n_outputs: int) -> np.ndarray:
    """
    He initialization for weights
	Output shape: (n_outputs, n_inputs)
	"""
    
    net_in = n_inputs
    net_out = n_outputs
    limit = np.sqrt(6. / (net_in + net_out))
    return rng.uniform(-limit, limit + 1e-5, size=(net_out, net_in))

def he_initialization_conv(n_inputs: int, n_outputs: int, kernel_height: int, kernel_width: int) -> np.ndarray:
    """
    He initialization for weights
    Output shape: (n_outputs, n_inputs, kernel_height, kernel_width)
    """

    net_in = n_inputs
    net_out = n_outputs
    limit = np.sqrt(6. / (net_in + net_out))
    return rng.uniform(-limit, limit + 1e-5, size=(net_out, net_in, kernel_height, kernel_width))


def init_biases(n_outputs: int) -> np.ndarray:
    """
    Bias initialization
    Output shape: (n_outputs, 1)
    """
    return rng.random((n_outputs, 1)) * 2 - 1


def get_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Generate positional encodings as described in "Attention is All You Need" paper.
    
    Args:
        seq_len: Length of the sequence
        d_model: Dimension of the model
    
    Returns:
        Positional encodings of shape (d_model, seq_len)
    """
    position = np.arange(seq_len).reshape(-1, 1)
    
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe.T