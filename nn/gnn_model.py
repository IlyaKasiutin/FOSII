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
    
    def train_step(self, node_features: np.ndarray, edge_features: np.ndarray,
                   adjacency_list: list, y_true: np.ndarray, loss_fn,
                   learning_rate: float = 0.001, optimizer=None) -> float:
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
        
        self.update_params(learning_rate, optimizer)
        self.zero_grad()
        
        return batch_loss
    
    def train(self, graphs: list, y_train: np.ndarray,
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001,
              loss_fn=None, optimizer=None, verbose: bool = True) -> dict:
        """
        Train the GNN model with batch training.
        
        Args:
            graphs: List of graphs, each graph is a dict with keys:
                    'node_features': np.ndarray of shape (num_nodes, node_features)
                    'edge_features': np.ndarray of shape (num_edges, edge_features)
                    'adjacency_list': list of lists
            y_train: Training targets of shape (num_samples, num_classes)
            epochs: Number of training epochs
            batch_size: Size of training batches
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
        
        num_samples = len(graphs)
        history = {
            'train_loss': [],
        }
        
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
                        target, loss_fn, learning_rate, optimizer
                    )
                    batch_loss += sample_loss
                
                batch_loss /= len(batch_graphs)  # Average loss for the batch
                total_loss += batch_loss
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            history['train_loss'].append(avg_train_loss)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.4f}")
        
        return history