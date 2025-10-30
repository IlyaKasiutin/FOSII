import numpy as np
from typing import List
from .module import Module
from .layers.convolution import Conv2d
from .layers.pooling import AvgPool2d
from .layers.linear import Linear
from .layers.tanh import Tanh
from .layers.sigmoid import Sigmoid
from tqdm import tqdm


class LeNet(Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        
        self.conv_layers = [
            Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            Sigmoid(),
            AvgPool2d(kernel_size=2, stride=2),
            
            Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            Sigmoid(),
            AvgPool2d(kernel_size=2, stride=2),
            
            Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
            Sigmoid()
        ]
        
        self.fc_layers = [
            Linear(120, 84),
            Sigmoid(),
            Linear(84, num_classes),
        ]
        
        self.all_layers = self.conv_layers + self.fc_layers
        
        self._modules = {}
        for i, layer in enumerate(self.conv_layers):
            self._modules[f"conv_layer_{i}"] = layer
        for i, layer in enumerate(self.fc_layers):
            self._modules[f"fc_layer_{i}"] = layer


    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.conv_layers:
            x = layer.forward(x)
        
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        x = x.T
        for layer in self.fc_layers:
            x = layer.forward(x)
        
        return x.T


    def backward(self, loss_grad: np.ndarray) -> None:
        output_grad = loss_grad.T
        
        for layer in reversed(self.fc_layers):
            output_grad = layer.backward(output_grad)
        
        batch_size = output_grad.shape[1]
        output_grad = output_grad.T.reshape(batch_size, -1, 1, 1)
        
        for layer in reversed(self.conv_layers):
            output_grad = layer.backward(output_grad)


    def update_params(self, learning_rate: float = 0.001) -> None:
        for layer in self.all_layers:
            if hasattr(layer, 'update_params'):
                layer.update_params(learning_rate)


    def zero_grad(self) -> None:
        for layer in self.all_layers:
            if hasattr(layer, 'zero_grad'):
                layer.zero_grad()
                
    def train_step(self, X: np.ndarray, y_true: np.ndarray, loss_fn, learning_rate: float = 0.001) -> float:
        y_pred = self.forward(X)
        
        loss = loss_fn.forward(y_pred.T, y_true.T)
        batch_loss = np.mean(loss)
        
        loss_grad = loss_fn.backward(y_pred.T, y_true.T)
        self.backward(loss_grad.T)
        
        self.update_params(learning_rate)
        
        self.zero_grad()
        
        return batch_loss
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001,
              loss_fn=None, verbose: bool = True) -> dict:

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
            
            total_loss = 0
            num_batches = 0
            
            for i in tqdm(range(0, num_samples, batch_size), total=int(num_samples / batch_size)):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                batch_loss = self.train_step(X_batch, y_batch, loss_fn, learning_rate)
                total_loss += batch_loss
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            history['train_loss'].append(avg_train_loss)
            
            val_loss = None
            val_accuracy = None
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val)
                
                val_loss_vals = loss_fn.forward(y_val_pred.T, y_val.T)
                val_loss = np.mean(val_loss_vals)
                history['val_loss'].append(val_loss)
                
                y_val_pred_classes = np.argmax(y_val_pred, axis=1)
                y_val_true_classes = np.argmax(y_val, axis=1)
                val_accuracy = np.mean(y_val_pred_classes == y_val_true_classes)
                history['val_accuracy'].append(val_accuracy)
            
            if verbose:
                if val_loss is not None and val_accuracy is not None:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}")
        
        return history
