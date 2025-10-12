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


def init_biases(n_outputs: int):
    """
    Bias initialization
    Output shape: (n_outputs, 1)
    """
    return rng.random((n_outputs, 1)) * 2 - 1