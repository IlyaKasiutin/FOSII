from ..module import Module
from ..layers.linear import Linear
from ..layers.relu import ReLU
import numpy as np
from typing import Dict


class GNNLayer(Module):
    def __init__(self, node_features: int, edge_features: int, out_features: int):
        """
        A simple GNN layer that performs message passing with edge features.
        
        Args:
            node_features: Number of input node features
            edge_features: Number of edge features
            out_features: Number of output node features
        """
        super().__init__()
        
        # Linear transformation for node features
        self.node_transform = Linear(node_features, out_features)
        
        # Linear transformation for edge features
        self.edge_transform = Linear(edge_features, out_features)
        
        # Linear transformation for aggregated messages
        self.message_transform = Linear(out_features, out_features)
        
        # Activation function
        self.activation = ReLU()
        
        # Store modules
        self._modules = {
            "node_transform": self.node_transform,
            "edge_transform": self.edge_transform,
            "message_transform": self.message_transform
        }
        
        # Cache for backward pass
        self._cache: Dict[str, np.ndarray] = {}
    
    def forward(self, node_features: np.ndarray, edge_features: np.ndarray, 
                adjacency_list: list) -> np.ndarray:
        """
        Forward pass of the GNN layer.
        
        Args:
            node_features: Node features of shape (num_nodes, node_features)
            edge_features: Edge features of shape (num_edges, edge_features)
            adjacency_list: List of lists, where adjacency_list[i] contains neighbors of node i
            
        Returns:
            Updated node features of shape (num_nodes, out_features)
        """
        num_nodes, node_feat_dim = node_features.shape
        num_edges, edge_feat_dim = edge_features.shape
        
        # Transform node features: (num_nodes, node_features) -> (node_features, num_nodes)
        node_features_t = node_features.T
        transformed_nodes = self.node_transform(node_features_t).T  # (num_nodes, out_features)
        
        # Transform edge features: (num_edges, edge_features) -> (edge_features, num_edges)
        edge_features_t = edge_features.T
        transformed_edges = self.edge_transform(edge_features_t).T  # (num_edges, out_features)
        
        # Message passing
        messages = np.zeros_like(transformed_nodes)  # (num_nodes, out_features)
        
        # For each edge, aggregate messages
        edge_idx = 0
        for i, neighbors in enumerate(adjacency_list):
            for j in neighbors:
                # Add edge feature to message
                messages[i] += transformed_edges[edge_idx]
                edge_idx += 1
        
        # Transform aggregated messages
        messages_t = messages.T  # (out_features, num_nodes)
        transformed_messages = self.message_transform(messages_t).T  # (num_nodes, out_features)
        
        # Combine node features with messages
        output = transformed_nodes + transformed_messages
        
        # Apply activation
        output = self.activation(output)
        
        # Cache for backward pass
        self._cache["node_features"] = node_features
        self._cache["edge_features"] = edge_features
        self._cache["adjacency_list"] = adjacency_list
        self._cache["transformed_nodes"] = transformed_nodes
        self._cache["transformed_edges"] = transformed_edges
        self._cache["messages"] = messages
        self._cache["output"] = output
        
        return output
    
    def backward(self, output_grad: np.ndarray) -> tuple:
        """
        Backward pass of the GNN layer.
        
        Args:
            output_grad: Gradient of loss w.r.t. output, shape (num_nodes, out_features)
            
        Returns:
            Tuple of gradients: (node_features_grad, edge_features_grad)
        """
        # Retrieve cached values
        node_features = self._cache["node_features"]
        edge_features = self._cache["edge_features"]
        adjacency_list = self._cache["adjacency_list"]
        transformed_nodes = self._cache["transformed_nodes"]
        transformed_edges = self._cache["transformed_edges"]
        messages = self._cache["messages"]
        output = self._cache["output"]
        
        num_nodes, node_feat_dim = node_features.shape
        num_edges, edge_feat_dim = edge_features.shape
        
        # Apply activation backward
        activation_grad = self.activation.backward(output_grad)  # (num_nodes, out_features)
        
        # Backward through combination: output = transformed_nodes + transformed_messages
        # Gradient is equally distributed to both operands
        transformed_nodes_grad = activation_grad  # (num_nodes, out_features)
        transformed_messages_grad = activation_grad  # (num_nodes, out_features)
        
        # Backward through message transformation: messages.T -> Linear -> .T -> transformed_messages
        # First transpose back: transformed_messages -> messages
        transformed_messages_grad_t = transformed_messages_grad.T  # (out_features, num_nodes)
        messages_grad_t = self.message_transform.backward(transformed_messages_grad_t)  # (out_features, num_nodes)
        messages_grad = messages_grad_t.T  # (num_nodes, out_features)
        
        # Backward through message aggregation
        transformed_edges_grad = np.zeros_like(transformed_edges)  # (num_edges, out_features)
        edge_idx = 0
        for i, neighbors in enumerate(adjacency_list):
            for j in neighbors:
                # Each edge contributes to the message of its source node
                transformed_edges_grad[edge_idx] = messages_grad[i]
                edge_idx += 1
        
        # Backward through edge feature transformation: edge_features.T -> Linear -> .T -> transformed_edges
        # First transpose back: transformed_edges -> edge_features
        transformed_edges_grad_t = transformed_edges_grad.T  # (out_features, num_edges)
        edge_features_grad_t = self.edge_transform.backward(transformed_edges_grad_t)  # (edge_feat_dim, num_edges)
        edge_features_grad = edge_features_grad_t.T  # (num_edges, edge_feat_dim)
        
        # Backward through node feature transformation: node_features.T -> Linear -> .T -> transformed_nodes
        # First transpose back: transformed_nodes -> node_features
        transformed_nodes_grad_t = transformed_nodes_grad.T  # (out_features, num_nodes)
        node_features_grad_t = self.node_transform.backward(transformed_nodes_grad_t)  # (node_feat_dim, num_nodes)
        node_features_grad = node_features_grad_t.T  # (num_nodes, node_feat_dim)
        
        return node_features_grad, edge_features_grad
    
    def update_params(self, learning_rate: float = 0.001, optimizer=None) -> None:
        """Update parameters of the layer."""
        if optimizer is not None:
            optimizer.step(self)
        else:
            self.node_transform.update_params(learning_rate)
            self.edge_transform.update_params(learning_rate)
            self.message_transform.update_params(learning_rate)
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.node_transform.zero_grad()
        self.edge_transform.zero_grad()
        self.message_transform.zero_grad()