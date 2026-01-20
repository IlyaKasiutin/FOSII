import numpy as np
from typing import Optional, Dict
from .module import Module
from .layers.linear import Linear
from .layers.relu import ReLU
from .layers.gnn_layer import GNNLayer
import tqdm


class GNN(Module):
    def __init__(self, node_features: int, edge_features: int, hidden_features: int,
                 num_classes: int, num_layers: int = 2):
        """
        A simple GNN model for graph classification or regression.
        
        Args:
            node_features: Number of input node features
            edge_features: Number of edge features
            hidden_features: Number of hidden features in GNN layers
            num_classes: Number of output classes
            num_layers: Number of GNN layers
        """
        super().__init__()
        
        self.layers = []
        in_features = node_features
        
        # Create GNN layers
        for i in range(num_layers):
            layer = GNNLayer(in_features, edge_features, hidden_features)
            self.layers.append(layer)
            self._modules[f"gnn_layer_{i}"] = layer
            in_features = hidden_features
        
        # Readout layer (simple sum pooling + MLP)
        self.readout = Linear(hidden_features, num_classes)
        self._modules["readout"] = self.readout
        
        # Cache for backward pass
        self._cache: Dict[str, list] = {
            "layer_outputs": []
        }
    
    def forward(self, node_features: np.ndarray, edge_features: np.ndarray, 
                adjacency_list: list) -> np.ndarray:
        """
        Forward pass of the GNN model.
        
        Args:
            node_features: Node features of shape (num_nodes, node_features)
            edge_features: Edge features of shape (num_edges, edge_features)
            adjacency_list: List of lists, where adjacency_list[i] contains neighbors of node i
            
        Returns:
            Graph-level prediction of shape (num_classes,)
        """
        # Clear cache
        self._cache["layer_outputs"].clear()
        
        # Forward through GNN layers
        x = node_features
        for layer in self.layers:
            x = layer(x, edge_features, adjacency_list)
            self._cache["layer_outputs"].append(x)
        
        # Readout: sum pooling
        graph_embedding = np.sum(x, axis=0)  # (hidden_features,)
        
        # Apply readout layer
        graph_embedding = graph_embedding.reshape(-1, 1)  # (hidden_features, 1)
        output = self.readout(graph_embedding)  # (num_classes, 1)
        
        return output.squeeze(axis=1)  # (num_classes,)
    
    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Backward pass of the GNN model.
        
        Args:
            loss_grad: Gradient of loss w.r.t. output, shape (num_classes,)
        """
        # Backward through readout layer
        loss_grad = loss_grad.reshape(-1, 1)  # (num_classes, 1)
        readout_grad = self.readout.backward(loss_grad)  # (hidden_features, 1)
        readout_grad = readout_grad.squeeze(axis=1)  # (hidden_features,)
        
        # Distribute gradient to all nodes (sum pooling backward)
        num_nodes = self._cache["layer_outputs"][-1].shape[0]
        node_grad = np.tile(readout_grad, (num_nodes, 1))  # (num_nodes, hidden_features)
        
        # Backward through GNN layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i > 0:
                prev_output = self._cache["layer_outputs"][i-1]
            else:
                prev_output = None  # Will use original node features from cache if needed
            
            node_grad, edge_grad = layer.backward(node_grad)
        
    
    def update_params(self, learning_rate: float = 0.001, optimizer=None) -> None:
        """Update parameters of all layers."""
        if optimizer is not None:
            optimizer.step(self)
        else:
            for layer in self.layers:
                layer.update_params(learning_rate)
            self.readout.update_params(learning_rate)
    
    def zero_grad(self) -> None:
        """Zero out gradients of all layers."""
        for layer in self.layers:
            layer.zero_grad()
        self.readout.zero_grad()
    
    def clip_gradients(self, max_norm: float = 1.0) -> None:
        """
        Clip gradients to prevent explosion.
        
        Args:
            max_norm: Maximum gradient norm
        """
        # Zero out NaN/Inf gradients first
        for layer in self.layers:
            if hasattr(layer, 'node_transform'):
                layer.node_transform.W_grad = np.where(
                    np.isfinite(layer.node_transform.W_grad),
                    layer.node_transform.W_grad, 0.0
                )
                layer.node_transform.bias_grad = np.where(
                    np.isfinite(layer.node_transform.bias_grad),
                    layer.node_transform.bias_grad, 0.0
                )
                layer.edge_transform.W_grad = np.where(
                    np.isfinite(layer.edge_transform.W_grad),
                    layer.edge_transform.W_grad, 0.0
                )
                layer.edge_transform.bias_grad = np.where(
                    np.isfinite(layer.edge_transform.bias_grad),
                    layer.edge_transform.bias_grad, 0.0
                )
                layer.message_transform.W_grad = np.where(
                    np.isfinite(layer.message_transform.W_grad),
                    layer.message_transform.W_grad, 0.0
                )
                layer.message_transform.bias_grad = np.where(
                    np.isfinite(layer.message_transform.bias_grad),
                    layer.message_transform.bias_grad, 0.0
                )
        if hasattr(self, 'readout'):
            self.readout.W_grad = np.where(
                np.isfinite(self.readout.W_grad),
                self.readout.W_grad, 0.0
            )
            self.readout.bias_grad = np.where(
                np.isfinite(self.readout.bias_grad),
                self.readout.bias_grad, 0.0
            )
        
        # Compute total norm
        total_norm = 0.0
        for layer in self.layers:
            if hasattr(layer, 'node_transform'):
                total_norm += np.sum(layer.node_transform.W_grad ** 2)
                total_norm += np.sum(layer.node_transform.bias_grad ** 2)
                total_norm += np.sum(layer.edge_transform.W_grad ** 2)
                total_norm += np.sum(layer.edge_transform.bias_grad ** 2)
                total_norm += np.sum(layer.message_transform.W_grad ** 2)
                total_norm += np.sum(layer.message_transform.bias_grad ** 2)
        if hasattr(self, 'readout'):
            total_norm += np.sum(self.readout.W_grad ** 2)
            total_norm += np.sum(self.readout.bias_grad ** 2)
        
        total_norm = np.sqrt(total_norm)
        
        # Clip if norm exceeds threshold
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-8)
            for layer in self.layers:
                if hasattr(layer, 'node_transform'):
                    layer.node_transform.W_grad *= clip_coef
                    layer.node_transform.bias_grad *= clip_coef
                    layer.edge_transform.W_grad *= clip_coef
                    layer.edge_transform.bias_grad *= clip_coef
                    layer.message_transform.W_grad *= clip_coef
                    layer.message_transform.bias_grad *= clip_coef
            if hasattr(self, 'readout'):
                self.readout.W_grad *= clip_coef
                self.readout.bias_grad *= clip_coef
    
    def train_step(self, node_features: np.ndarray, edge_features: np.ndarray,
                   adjacency_list: list, y_true: np.ndarray, loss_fn,
                   learning_rate: float = 0.001, optimizer=None, gradient_clip: float = None) -> float:
        """
        Perform a single training step.
        
        Args:
            node_features: Node features of shape (num_nodes, node_features)
            edge_features: Edge features of shape (num_edges, edge_features)
            adjacency_list: List of lists, where adjacency_list[i] contains neighbors of node i
            y_true: True labels of shape (num_classes,) for classification or (num_outputs,) for regression
            loss_fn: Loss function
            learning_rate: Learning rate
            optimizer: Optional optimizer
            
        Returns:
            Batch loss
        """
        y_pred = self.forward(node_features, edge_features, adjacency_list)
        
        # Reshape for loss function: (num_outputs, batch_size) where batch_size=1
        y_pred_reshaped = y_pred.reshape(-1, 1)  # (num_outputs, 1)
        y_true_reshaped = y_true.reshape(-1, 1)  # (num_outputs, 1)
        
        loss_vals = loss_fn.forward(y_pred_reshaped, y_true_reshaped)
        batch_loss = float(np.mean(loss_vals))
        
        loss_grad = loss_fn.backward(y_pred_reshaped, y_true_reshaped).squeeze(axis=1)
        self.backward(loss_grad)
        
        # Clip gradients if specified
        if gradient_clip is not None:
            self.clip_gradients(gradient_clip)
        
        self.update_params(learning_rate, optimizer)
        self.zero_grad()
        
        return batch_loss
    
    def train(self, graphs: list, y_train: np.ndarray,
              graphs_val: list = None, y_val: np.ndarray = None,
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001,
              loss_fn=None, optimizer=None, verbose: bool = True, gradient_clip: float = None) -> dict:
        """
        Train the GNN model with batch training.
        
        Args:
            graphs: List of graphs, each graph is a dict with keys:
                    'node_features': np.ndarray of shape (num_nodes, node_features)
                    'edge_features': np.ndarray of shape (num_edges, edge_features)
                    'adjacency_list': list of lists
            y_train: Training targets of shape (num_samples, num_classes)
            graphs_val: Optional validation graphs (same format as graphs)
            y_val: Optional validation targets of shape (num_val_samples, num_classes)
            epochs: Number of training epochs
            batch_size: Size of training batches
            learning_rate: Learning rate
            loss_fn: Loss function (defaults to CrossEntropy)
            optimizer: Optimizer (optional)
            verbose: Whether to print training progress
            gradient_clip: Maximum gradient norm for clipping (None to disable)
            
        Returns:
            Dictionary with training history
        """
        if loss_fn is None:
            from .losses.cross_entropy import CrossEntropy
            loss_fn = CrossEntropy()
        
        num_samples = len(graphs)
        history = {
            'train_loss': [],
        }
        if graphs_val is not None and y_val is not None:
            history['val_loss'] = []
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(num_samples)
            graphs_shuffled = [graphs[i] for i in indices]
            y_train_shuffled = y_train[indices]
            
            total_loss = 0.0
            num_batches = 0
            
            # Process in batches
            for start in tqdm.tqdm(range(0, num_samples, batch_size), total=(num_samples + batch_size - 1) // batch_size, disable=not verbose):
                end = min(start + batch_size, num_samples)
                batch_graphs = graphs_shuffled[start:end]
                batch_targets = y_train_shuffled[start:end]
                
                batch_loss = 0.0
                # Process each graph in the batch
                for graph, target in zip(batch_graphs, batch_targets):
                    sample_loss = self.train_step(
                        graph['node_features'], graph['edge_features'], graph['adjacency_list'],
                        target, loss_fn, learning_rate, optimizer, gradient_clip=gradient_clip
                    )
                    batch_loss += sample_loss
                
                batch_loss /= len(batch_graphs)  # Average loss for the batch
                total_loss += batch_loss
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            history['train_loss'].append(avg_train_loss)
            
            # Compute validation loss if validation data is provided
            val_loss = None
            if graphs_val is not None and y_val is not None:
                val_losses = []
                for graph, target in zip(graphs_val, y_val):
                    output = self.forward(
                        graph['node_features'],
                        graph['edge_features'],
                        graph['adjacency_list']
                    )
                    y_pred = output.reshape(-1, 1)
                    y_true = target.reshape(-1, 1)
                    loss_vals = loss_fn.forward(y_pred, y_true)
                    val_losses.append(float(np.mean(loss_vals)))
                
                val_loss = np.mean(val_losses)
                history['val_loss'].append(val_loss)
            
            if verbose:
                if val_loss is not None:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f} - val_loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}")
        
        return history