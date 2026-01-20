import numpy as np
from .gnn_model import GNN
from .losses.cross_entropy import CrossEntropy


def create_sample_graph(num_nodes=5, node_features=3, edge_features=2, graph_type=0):
    """
    Create a sample graph for testing.
    
    Args:
        num_nodes: Number of nodes in the graph
        node_features: Number of node features
        edge_features: Number of edge features
        graph_type: Type of graph to create (0 or 1) for binary classification
        
    Returns:
        Dictionary with graph data
    """
    # Create node features based on graph type
    if graph_type == 0:
        # Graph type 0: nodes with higher first feature
        node_feats = np.random.randn(num_nodes, node_features)
        node_feats[:, 0] += 2  # Increase first feature
    else:
        # Graph type 1: nodes with higher second feature
        node_feats = np.random.randn(num_nodes, node_features)
        node_feats[:, 1] += 2  # Increase second feature
    
    # Create a simple adjacency list (ring graph)
    adjacency_list = []
    for i in range(num_nodes):
        neighbors = [(i - 1) % num_nodes, (i + 1) % num_nodes]
        adjacency_list.append(neighbors)
    
    # Create edge features (2 edges per node in ring graph)
    num_edges = num_nodes * 2
    edge_feats = np.random.randn(num_edges, edge_features)
    
    return {
        'node_features': node_feats,
        'edge_features': edge_feats,
        'adjacency_list': adjacency_list
    }


def example_gnn():
    """Example of using the GNN model."""
    print("Creating sample graphs...")
    
    # Create sample graphs
    graphs = []
    for i in range(10):
        graph_type = i % 2  # Alternate between two graph types
        graph = create_sample_graph(num_nodes=5, node_features=3, edge_features=2, graph_type=graph_type)
        graphs.append(graph)
    
    # Create sample labels (one-hot encoded)
    y_train = np.eye(2)[np.random.choice(2, 10)].astype(np.float32)
    
    print(f"Created {len(graphs)} sample graphs")
    print(f"Node features shape: {graphs[0]['node_features'].shape}")
    print(f"Edge features shape: {graphs[0]['edge_features'].shape}")
    print(f"Adjacency list: {graphs[0]['adjacency_list']}")
    print(f"Labels shape: {y_train.shape}")
    
    # Create GNN model
    model = GNN(
        node_features=3,
        edge_features=2,
        hidden_features=8,
        num_classes=2,  # Changed to 2 for binary classification
        num_layers=2
    )
    
    print("\nModel created successfully!")
    print(f"Number of GNN layers: 2")
    print(f"Hidden features: 8")
    print(f"Number of classes: 2")
    
    # Forward pass on a single graph
    print("\nTesting forward pass...")
    sample_graph = graphs[0]
    output = model.forward(
        sample_graph['node_features'],
        sample_graph['edge_features'],
        sample_graph['adjacency_list']
    )
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    # Create loss function
    loss_fn = CrossEntropy()
    
    # Compute loss
    y_true = y_train[0]
    # Reshape for CrossEntropy: (num_classes, batch_size) where batch_size=1
    y_pred_reshaped = output.reshape(-1, 1)  # (num_classes, 1)
    y_true_reshaped = y_true.reshape(-1, 1)  # (num_classes, 1)
    
    loss_vals = loss_fn.forward(y_pred_reshaped, y_true_reshaped)
    loss = float(np.mean(loss_vals))
    print(f"\nLoss: {loss:.4f}")
    
    # Backward pass
    loss_grad = loss_fn.backward(y_pred_reshaped, y_true_reshaped).squeeze(axis=1)
    model.backward(loss_grad)
    print("Backward pass completed successfully!")
    
    # Update parameters
    model.update_params(learning_rate=0.001)
    print("Parameters updated successfully!")
    
    # Zero gradients
    model.zero_grad()
    print("Gradients zeroed successfully!")
    
    print("\nGNN basic example completed successfully!")


def example_gnn_training():
    """Example of training the GNN model."""
    print("Creating sample graphs for training...")
    
    # Create sample graphs with more distinct patterns
    graphs = []
    labels = []
    for i in range(20):
        # Alternate between two types of graphs
        graph_type = i % 2
        graph = create_sample_graph(num_nodes=4, node_features=3, edge_features=2, graph_type=graph_type)
        graphs.append(graph)
        labels.append(graph_type)
    
    # Create sample labels (one-hot encoded)
    y_train = np.eye(2)[labels].astype(np.float32)
    
    print(f"Created {len(graphs)} sample graphs for training")
    print(f"Graph types: {labels}")
    
    # Create GNN model
    model = GNN(
        node_features=3,
        edge_features=2,
        hidden_features=8,
        num_classes=2,  # Binary classification
        num_layers=2
    )
    
    # Create loss function
    loss_fn = CrossEntropy()
    
    # Train the model
    print("\nTraining GNN model...")
    history = model.train(
        graphs, y_train,
        epochs=10,  # Increase epochs to see more training progress
        learning_rate=0.01,  # Increase learning rate
        loss_fn=loss_fn,
        verbose=True
    )
    
    print("\nTraining completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    
    # Test forward pass after training
    test_graph = graphs[0]
    test_output = model.forward(
        test_graph['node_features'],
        test_graph['edge_features'],
        test_graph['adjacency_list']
    )
    print(f"\nTest output shape: {test_output.shape}")
    print(f"Test output values: {test_output}")
    print("GNN training example completed successfully!")


if __name__ == "__main__":
    example_gnn()
    print("\n" + "="*50 + "\n")
    example_gnn_training()