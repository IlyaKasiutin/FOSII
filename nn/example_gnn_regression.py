import numpy as np
from .gnn_model import GNN
from .losses.mse import MSE


def create_sample_graph_for_regression(num_nodes=5, node_features=3, edge_features=2, target_type="scalar"):
    node_feats = np.random.randn(num_nodes, node_features)
    
    for i in range(num_nodes):
        node_feats[i, 0] += i * 0.5  # Increase first feature with node index
    
    adjacency_list = []
    for i in range(num_nodes):
        neighbors = [(i - 1) % num_nodes, (i + 1) % num_nodes]
        adjacency_list.append(neighbors)
    
    num_edges = num_nodes * 2
    edge_feats = np.random.randn(num_edges, edge_features)

    if target_type == "scalar":
        target = np.array([np.sum(node_feats[:, 0])])
    else:
        target = np.sum(node_feats, axis=0)
    
    return {
        'node_features': node_feats,
        'edge_features': edge_feats,
        'adjacency_list': adjacency_list
    }, target


def example_gnn_regression():
    graphs = []
    targets = []
    for i in range(10):
        graph, target = create_sample_graph_for_regression(
            num_nodes=5, node_features=3, edge_features=2, target_type="scalar")
        graphs.append(graph)
        targets.append(target)
    
    y_train = np.array(targets)
    
    print(f"Created {len(graphs)} sample graphs")
    print(f"Node features shape: {graphs[0]['node_features'].shape}")
    print(f"Edge features shape: {graphs[0]['edge_features'].shape}")
    print(f"Adjacency list: {graphs[0]['adjacency_list']}")
    print(f"Targets shape: {y_train.shape}")
    print(f"Sample targets: {y_train[:3].flatten()}")
    
    model = GNN(
        node_features=3,
        edge_features=2,
        hidden_features=8,
        num_classes=1,
        num_layers=2
    )
    
    print("\nModel created successfully!")
    print(f"Number of GNN layers: 2")
    print(f"Hidden features: 8")
    print(f"Number of outputs: 1 (scalar regression)")
    
    print("\nTesting forward pass...")
    sample_graph = graphs[0]
    output = model.forward(
        sample_graph['node_features'],
        sample_graph['edge_features'],
        sample_graph['adjacency_list']
    )
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    loss_fn = MSE()
    
    y_true = y_train[0]
    y_pred_reshaped = output.reshape(-1, 1)  # (1, 1) for scalar
    y_true_reshaped = y_true.reshape(-1, 1)  # (1, 1) for scalar
    
    loss_vals = loss_fn.forward(y_pred_reshaped, y_true_reshaped)
    loss = float(np.mean(loss_vals))
    print(f"\nLoss: {loss:.4f}")
    
    loss_grad = loss_fn.backward(y_pred_reshaped, y_true_reshaped).squeeze(axis=1)
    model.backward(loss_grad)
    
    model.update_params(learning_rate=0.001)
    
    model.zero_grad()


def example_gnn_regression_training():
    graphs = []
    targets = []
    for i in range(20):
        graph, target = create_sample_graph_for_regression(
            num_nodes=4, node_features=3, edge_features=2, target_type="scalar")
        graphs.append(graph)
        targets.append(target)
    
    y_train = np.array(targets)
    
    print(f"Created {len(graphs)} sample graphs for training")
    print(f"Sample targets: {y_train[:5].flatten()}")
    
    model = GNN(
        node_features=3,
        edge_features=2,
        hidden_features=8,
        num_classes=1,
        num_layers=2
    )
    
    loss_fn = MSE()
    
    print("\nTraining GNN model for regression...")
    history = model.train(
        graphs, y_train,
        epochs=10,
        learning_rate=0.01,
        loss_fn=loss_fn,
        verbose=True
    )
    
    print("\nTraining completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    
    test_graph = graphs[0]
    test_output = model.forward(
        test_graph['node_features'],
        test_graph['edge_features'],
        test_graph['adjacency_list']
    )
    print(f"\nTest output shape: {test_output.shape}")
    print(f"Test output values: {test_output}")
    print(f"True target: {y_train[0]}")
    print("GNN regression training example completed successfully!")


if __name__ == "__main__":
    example_gnn_regression()
    print("\n" + "="*50 + "\n")
    example_gnn_regression_training()