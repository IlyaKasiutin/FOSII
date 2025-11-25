import numpy as np
from typing import Optional, Dict
from .module import Module
from .layers.transformer_block import TransformerBlock
from .layers.linear import Linear
import tqdm


class Transformer(Module):
    """
    Transformer model for sequence-to-sequence or sequence-to-one tasks.
    Follows the same pattern as RNN/GRU/LSTM models in this codebase.
    """
    
    def __init__(self, n_inputs: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, n_outputs: int, max_seq_len: int = 100, dropout: float = 0.0):
        """
        Initialize Transformer model.
        
        Args:
            n_inputs: Input feature dimension
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            d_ff: Feed-forward dimension
            n_outputs: Output dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate (not implemented yet, kept for API compatibility)
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_inputs = n_inputs
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.n_outputs = n_outputs
        self.max_seq_len = max_seq_len
        
        # Input projection to d_model
        self.input_proj = Linear(n_inputs, d_model)
        
        # Stack of transformer blocks
        self.blocks = []
        for i in range(n_layers):
            block = TransformerBlock(d_model, n_heads, d_ff, dropout)
            self.blocks.append(block)
            self._modules[f"block_{i}"] = block
        
        self._modules["input_proj"] = self.input_proj
        
        # Output projection (for many-to-one: take last token or pool)
        self.output_proj = Linear(d_model, n_outputs)
        self._modules["output_proj"] = self.output_proj
        
        # Cache for backward pass
        self._cache: Dict[str, list] = {
            "inputs": [],
            "block_outputs": []
        }
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of transformer.
        
        Args:
            x: Input of shape (seq_len, n_inputs) for single sequence
               or (batch, seq_len, n_inputs) for batch
        
        Returns:
            Output logits of shape (n_outputs,) for single sequence
               or (batch, n_outputs) for batch
        """
        # Handle single sequence: (seq_len, n_inputs)
        if x.ndim == 2:
            seq_len, _ = x.shape
            x = x.T  # (n_inputs, seq_len)
            
            # Project input to d_model
            x = self.input_proj(x)  # (d_model, seq_len)
            
            # Store for backward
            self._cache["inputs"] = [x]
            self._cache["block_outputs"] = []
            
            # Pass through transformer blocks
            for block in self.blocks:
                x = block(x)  # (d_model, seq_len)
                self._cache["block_outputs"].append(x.copy())
            
            # For many-to-one: take the last token representation
            # x is (d_model, seq_len), take last column
            last_token = x[:, -1:]  # (d_model, 1)
            
            # Project to output
            output = self.output_proj(last_token)  # (n_outputs, 1)
            
            return output.squeeze(axis=1)  # (n_outputs,)
        
        else:
            # Batch processing: (batch, seq_len, n_inputs)
            batch_size, seq_len, _ = x.shape
            outputs = []
            
            for i in range(batch_size):
                x_seq = x[i].T  # (n_inputs, seq_len)
                x_seq = self.input_proj(x_seq)
                
                for block in self.blocks:
                    x_seq = block(x_seq)
                
                last_token = x_seq[:, -1:]
                output = self.output_proj(last_token)
                outputs.append(output.squeeze(axis=1))
            
            return np.array(outputs).T  # (n_outputs, batch) -> transpose to (batch, n_outputs)
    
    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Backward pass of transformer.
        
        Args:
            loss_grad: Gradient from loss function, shape (n_outputs,) for single sequence
                       or (batch, n_outputs) for batch
        """
        # Handle single sequence (assume cache is set from forward)
        if loss_grad.ndim == 1:
            loss_grad = loss_grad.reshape(-1, 1)  # (n_outputs, 1)
        
        # Backward through output projection
        doutput = self.output_proj.backward(loss_grad)  # (d_model, 1)
        
        # Expand to full sequence length (only last token has gradient)
        # Get sequence length from cached block outputs
        if len(self._cache["block_outputs"]) > 0:
            seq_len = self._cache["block_outputs"][-1].shape[1]
        else:
            # Fallback: assume we can infer from input cache
            seq_len = self._cache["inputs"][0].shape[1] if len(self._cache["inputs"]) > 0 else 1
        
        dblock_output = np.zeros((self.d_model, seq_len))
        dblock_output[:, -1:] = doutput  # Only last token has gradient
        
        # Backward through transformer blocks in reverse
        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            dblock_output = block.backward(dblock_output)
        
        # Backward through input projection
        self.input_proj.backward(dblock_output)
    
    def update_params(self, learning_rate: float = 0.001, optimizer=None) -> None:
        """Update all parameters."""
        if optimizer is not None:
            optimizer.step(self)
            return
        
        # Update input projection
        if hasattr(self.input_proj, "W_grad"):
            self.input_proj.update_params(learning_rate)
        
        # Update transformer blocks
        for block in self.blocks:
            block.update_params(learning_rate)
        
        # Update output projection
        if hasattr(self.output_proj, "W_grad"):
            self.output_proj.update_params(learning_rate)
    
    def zero_grad(self) -> None:
        """Zero out all gradients."""
        if hasattr(self.input_proj, "W_grad"):
            self.input_proj.W_grad.fill(0)
            self.input_proj.bias_grad.fill(0)
        
        for block in self.blocks:
            block.zero_grad()
        
        if hasattr(self.output_proj, "W_grad"):
            self.output_proj.W_grad.fill(0)
            self.output_proj.bias_grad.fill(0)
    
    def train_step(self, X: np.ndarray, y_true: np.ndarray, loss_fn,
                   learning_rate: float = 0.001, optimizer=None) -> float:
        """
        Run a single optimization step on a sequence or batch.
        
        Args:
            X: (seq_len, n_inputs) for single sequence or (batch, seq_len, n_inputs) for batch
            y_true: (n_outputs,) one-hot for CE or targets for regression
                   or (batch, n_outputs) for batch
        """
        batch_loss = 0
        
        # Handle single sequence
        if X.ndim == 2:
            y_pred = self.forward(X)  # (n_outputs,)
            
            # Align with patterns in other models
            loss_vals = loss_fn.forward(y_pred.T, y_true.T)
            batch_loss = float(np.mean(loss_vals))
            
            loss_grad = loss_fn.backward(y_pred.T, y_true.T).T  # (n_outputs,)
            self.backward(loss_grad)
        else:
            # Batch processing
            batch_size = X.shape[0]
            for i in range(batch_size):
                cur_x = X[i]  # (seq_len, n_inputs)
                cur_y = y_true[i]  # (n_outputs,)
                
                y_pred = self.forward(cur_x)  # (n_outputs,)
                
                loss_vals = loss_fn.forward(y_pred.T, cur_y.T)
                batch_loss += float(np.mean(loss_vals))
                
                loss_grad = loss_fn.backward(y_pred.T, cur_y.T).T  # (n_outputs,)
                self.backward(loss_grad)
        
        self.update_params(learning_rate, optimizer)
        self.zero_grad()
        
        return batch_loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001,
              loss_fn=None, optimizer=None, verbose: bool = True) -> dict:
        """
        Train the transformer model.
        
        Args:
            X_train: Training sequences, shape (num_samples, seq_len, n_inputs)
            y_train: Training targets, shape (num_samples, n_outputs)
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            loss_fn: Loss function (defaults to CrossEntropy)
            optimizer: Optimizer (optional)
            verbose: Whether to print training progress
        
        Returns:
            Dictionary with training history
        """
        if loss_fn is None:
            from .losses.cross_entropy import CrossEntropy
            loss_fn = CrossEntropy()
        
        num_samples = X_train.shape[0]
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            total_loss = 0.0
            
            for start in tqdm.tqdm(range(0, num_samples, batch_size), total=(num_samples / batch_size)):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                
                batch_loss = self.train_step(
                    X_batch, y_batch, loss_fn, learning_rate, optimizer
                )
                total_loss += batch_loss
            
            avg_train_loss = total_loss / num_samples
            history['train_loss'].append(avg_train_loss)
            
            val_loss = None
            val_accuracy = None
            if X_val is not None and y_val is not None:
                val_loss = 0
                for i in range(X_val.shape[0]):
                    cur_x = X_val[i]
                    cur_y = y_val[i]
                    y_pred = self.forward(cur_x)
                    loss_vals = loss_fn.forward(y_pred.T, cur_y.T)
                    val_loss += np.mean(loss_vals)
                
                val_loss /= X_val.shape[0]
                history['val_loss'].append(val_loss)
                
                # Compute accuracy if applicable
                try:
                    y_val_pred = []
                    for i in range(X_val.shape[0]):
                        y_val_pred.append(self.forward(X_val[i]))
                    y_val_pred = np.array(y_val_pred)
                    
                    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
                    y_val_true_classes = np.argmax(y_val, axis=1)
                    val_accuracy = float(np.mean(y_val_pred_classes == y_val_true_classes))
                    history['val_accuracy'].append(val_accuracy)
                except Exception:
                    pass
            
            if verbose:
                if val_loss is not None and val_accuracy is not None:
                    print(
                        f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} "
                        f"- val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
                    )
                elif val_loss is not None:
                    print(
                        f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} "
                        f"- val_loss: {val_loss:.4f}"
                    )
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}")
        
        return history

