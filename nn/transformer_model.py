import numpy as np
from typing import Optional, Dict
from .module import Module
from .layers.transformer_block import TransformerBlock
from .layers.linear import Linear
from .utils import get_positional_encoding
import tqdm


class Transformer(Module):
    def __init__(self, n_inputs: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, n_outputs: int, max_seq_len: int = 100, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_inputs = n_inputs
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.n_outputs = n_outputs
        
        self.input_proj = Linear(n_inputs, d_model)
        
        self.blocks = []
        for i in range(n_layers):
            block = TransformerBlock(d_model, n_heads, d_ff, dropout)
            self.blocks.append(block)
            self._modules[f"block_{i}"] = block
        
        self._modules["input_proj"] = self.input_proj
        
        self.output_proj = Linear(d_model, n_outputs)
        self._modules["output_proj"] = self.output_proj
        
        self._cache: Dict[str, list] = {
            "inputs": [],
            "block_outputs": []
        }
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            seq_len, _ = x.shape
            x = x.T  # (n_inputs, seq_len)
            
            projected_input = self.input_proj(x)  # (d_model, seq_len)
            
            pos_enc = get_positional_encoding(seq_len, self.d_model)  # (d_model, seq_len)
            x = projected_input + pos_enc  # (d_model, seq_len)
            # x = projected_input
            
            self._cache["inputs"] = [projected_input.copy()]
            self._cache["block_outputs"] = []
            
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
            batch_size, seq_len, _ = x.shape
            outputs = []
            
            pos_enc = get_positional_encoding(seq_len, self.d_model)  # (d_model, seq_len)
            
            for i in range(batch_size):
                x_seq = x[i].T  # (n_inputs, seq_len)
                projected_input = self.input_proj(x_seq)
                x_seq = projected_input + pos_enc
                # x_seq = projected_input
                
                for block in self.blocks:
                    x_seq = block(x_seq)
                
                last_token = x_seq[:, -1:]
                output = self.output_proj(last_token)
                outputs.append(output.squeeze(axis=1))
            
            return np.array(outputs).T  # (n_outputs, batch) -> transpose to (batch, n_outputs)
    
    def backward(self, loss_grad: np.ndarray) -> None:
        if loss_grad.ndim == 1:
            loss_grad = loss_grad.reshape(-1, 1)  # (n_outputs, 1)
        
        doutput = self.output_proj.backward(loss_grad)  # (d_model, 1)
        
        if len(self._cache["block_outputs"]) > 0:
            seq_len = self._cache["block_outputs"][-1].shape[1]
        else:
            seq_len = self._cache["inputs"][0].shape[1] if len(self._cache["inputs"]) > 0 else 1
        
        dblock_output = np.zeros((self.d_model, seq_len))
        dblock_output[:, -1:] = doutput  # Only last token has gradient
        
        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            dblock_output = block.backward(dblock_output)
        
        self.input_proj.backward(dblock_output)
    
    def update_params(self, learning_rate: float = 0.001, optimizer=None) -> None:
        """Update all parameters."""
        if optimizer is not None:
            optimizer.step(self)
            return
        
        if hasattr(self.input_proj, "W_grad"):
            self.input_proj.update_params(learning_rate)
        
        for block in self.blocks:
            block.update_params(learning_rate)
        
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
        batch_loss = 0
        
        if X.ndim == 2:
            y_pred = self.forward(X)  # (n_outputs,)
            
            loss_vals = loss_fn.forward(y_pred.T, y_true.T)
            batch_loss = float(np.mean(loss_vals))
            
            loss_grad = loss_fn.backward(y_pred.T, y_true.T).T  # (n_outputs,)
            self.backward(loss_grad)
        else:
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

