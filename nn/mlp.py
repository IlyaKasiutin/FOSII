import numpy as np
from typing import List
from .module import Module
from .layers.linear import Linear
from .layers.sigmoid import Sigmoid
from .losses.mse import MSE


class MLP(Module):
    def __init__(self, dims: List[int]):
        """
        Multi-Layer Perceptron implementation in PyTorch style
        
        Args:
            dims: List of layer dimensions [input_dim, hidden_dim1, ..., output_dim]
        """
        super().__init__()
        self.layers = []
        
        # Create layers
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))
            # Add activation function except for the output layer
            if i < len(dims) - 2:
                self.layers.append(Sigmoid())
                
        self._modules = {f"layer_{i}": layer for i, layer in enumerate(self.layers)}
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of the MLP
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Transpose to match layer expectations (features, batch)
        result = x.T
        
        # Forward through all layers
        for layer in self.layers:
            result = layer(result)
            
        # Transpose back to (batch, features)
        return result.T
    
    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Backward pass of the MLP
        
        Args:
            loss_grad: Gradient from loss function of shape (batch_size, output_dim)
        """
        # Transpose to match layer expectations (features, batch)
        output_grad = loss_grad  # (batch, features)
        
        # Backward through all layers in reverse order
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)
    
    def update_params(self, learning_rate: float = 0.001, optimizer=None) -> None:
        """
        Update parameters of all layers
        
        Args:
            learning_rate: Learning rate for parameter updates (used if optimizer is None)
            optimizer: Optional optimizer instance (SGD, Adam, etc.). If None, uses SGD with learning_rate
        """
        if optimizer is not None:
            optimizer.step(self)
        else:
            # Backward compatibility: use simple SGD
            for layer in self.layers:
                if hasattr(layer, 'update_params'):
                    layer.update_params(learning_rate)
    
    def zero_grad(self) -> None:
        """
        Zero out gradients of all layers
        """
        for layer in self.layers:
            if hasattr(layer, 'W_grad'):
                layer.W_grad.fill(0)
            if hasattr(layer, 'bias_grad'):
                layer.bias_grad.fill(0)

    def train_step(self, X: np.ndarray, y_true: np.ndarray, loss_fn, learning_rate: float = 0.001, optimizer=None) -> float:
        """
        Perform a single training step: forward, loss, backward, update, zero_grad
        """
        y_pred = self.forward(X)
        loss_vals = loss_fn.forward(y_pred, y_true)
        batch_loss = np.mean(loss_vals)
        
        loss_grad = loss_fn.backward(y_pred, y_true).T  # (n_out, batch)
        self.backward(loss_grad)
        
        self.update_params(learning_rate, optimizer)
        self.zero_grad()
        
        return batch_loss

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001,
              loss_fn=None, optimizer=None, verbose: bool = True) -> dict:
        """
        Train the MLP model using mini-batch gradient descent style loop.
        """
        if loss_fn is None:
            loss_fn = MSE()
        
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
            num_batches = 0
            
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                
                batch_loss = self.train_step(X_batch, y_batch, loss_fn, learning_rate, optimizer)
                total_loss += batch_loss
                num_batches += 1
            
            avg_train_loss = total_loss / max(1, num_batches)
            history['train_loss'].append(avg_train_loss)
            
            val_loss = None
            val_accuracy = None
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                val_loss_vals = loss_fn.forward(y_val_pred, y_val)
                val_loss = float(np.mean(val_loss_vals))
                history['val_loss'].append(val_loss)
                
                # If labels are one-hot, compute accuracy
                try:
                    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
                    y_val_true_classes = y_val
                    val_accuracy = float(np.mean(y_val_pred_classes == y_val_true_classes))
                    history['val_accuracy'].append(val_accuracy)
                except Exception:
                    # For regression targets this will fail; skip accuracy
                    pass
            
            if verbose:
                if val_loss is not None and (len(history['val_accuracy']) == len(history['val_loss'])):
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy if val_accuracy is not None else 0:.4f}")
                elif val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}")
        
        return history
