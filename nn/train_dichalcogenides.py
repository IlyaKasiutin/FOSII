"""
Training pipeline for GNN model on dichalcogenides dataset.
Predicts band gap from crystal structures.
"""

import os
import json
import csv
import numpy as np
from pymatgen.core import Structure
from typing import List, Dict, Tuple
import tqdm

from .gnn_model import GNN
from .losses.mse import MSE


def structure_to_graph(structure: Structure, cutoff_radius: float = 5.0) -> Dict:
    """
    Convert a pymatgen Structure to graph format for GNN.
    
    Args:
        structure: pymatgen Structure object
        cutoff_radius: Maximum distance (Å) for considering neighbors
        
    Returns:
        Dictionary with keys:
            - 'node_features': np.ndarray of shape (num_nodes, node_features)
            - 'edge_features': np.ndarray of shape (num_edges, edge_features)
            - 'adjacency_list': list of lists, where adjacency_list[i] contains neighbors of node i
    """
    num_nodes = len(structure)
    
    # 1. Node features (per atom)
    node_features = []
    for site in structure:
        elem = site.specie
        # Common features for GNNs:
        features = [
            elem.Z,  # Atomic number
            elem.atomic_mass,
            elem.row,  # Period
            elem.group,  # Group
            elem.atomic_radius if elem.atomic_radius else 0.0,
            # Add more as needed
        ]
        node_features.append(features)
    
    node_features = np.array(node_features, dtype=np.float64)
    
    # 2. Build adjacency list and edge features
    adjacency_list = [[] for _ in range(num_nodes)]
    edge_features = []
    
    for i, site in enumerate(structure):
        neighbors = structure.get_neighbors(site, r=cutoff_radius)
        for neighbor, distance, idx, _ in neighbors:
            # Add neighbor to adjacency list
            adjacency_list[i].append(idx)
            # Create edge features
            edge_feat = [
                distance,
                site.specie.Z,  # Source atom Z
                neighbor.specie.Z,  # Target atom Z
            ]
            edge_features.append(edge_feat)
    
    edge_features = np.array(edge_features, dtype=np.float64)
    
    return {
        'node_features': node_features,
        'edge_features': edge_features,
        'adjacency_list': adjacency_list
    }


def normalize_features(graphs: List[Dict], node_mean: np.ndarray = None, 
                       node_std: np.ndarray = None, edge_mean: np.ndarray = None,
                       edge_std: np.ndarray = None) -> Tuple[List[Dict], Dict]:
    """
    Normalize node and edge features across all graphs.
    
    Args:
        graphs: List of graph dictionaries
        node_mean: Precomputed mean for node features (if None, computed from graphs)
        node_std: Precomputed std for node features (if None, computed from graphs)
        edge_mean: Precomputed mean for edge features (if None, computed from graphs)
        edge_std: Precomputed std for edge features (if None, computed from graphs)
        
    Returns:
        Tuple of (normalized_graphs, normalization_stats) where normalization_stats
        contains 'node_mean', 'node_std', 'edge_mean', 'edge_std'
    """
    # Collect all features to compute statistics
    if node_mean is None or node_std is None:
        all_node_features = []
        for graph in graphs:
            all_node_features.append(graph['node_features'])
        all_node_features = np.vstack(all_node_features)
        node_mean = np.mean(all_node_features, axis=0, keepdims=True)
        node_std = np.std(all_node_features, axis=0, keepdims=True)
        # Avoid division by zero
        node_std = np.where(node_std < 1e-8, 1.0, node_std)
    
    if edge_mean is None or edge_std is None:
        all_edge_features = []
        for graph in graphs:
            if len(graph['edge_features']) > 0:
                all_edge_features.append(graph['edge_features'])
        if len(all_edge_features) > 0:
            all_edge_features = np.vstack(all_edge_features)
            edge_mean = np.mean(all_edge_features, axis=0, keepdims=True)
            edge_std = np.std(all_edge_features, axis=0, keepdims=True)
            # Avoid division by zero
            edge_std = np.where(edge_std < 1e-8, 1.0, edge_std)
        else:
            edge_mean = np.zeros((1, graphs[0]['edge_features'].shape[1]))
            edge_std = np.ones((1, graphs[0]['edge_features'].shape[1]))
    
    # Normalize graphs
    normalized_graphs = []
    for graph in graphs:
        normalized_graph = {
            'node_features': (graph['node_features'] - node_mean) / node_std,
            'adjacency_list': graph['adjacency_list']
        }
        # Normalize edge features if they exist
        if len(graph['edge_features']) > 0:
            normalized_graph['edge_features'] = (graph['edge_features'] - edge_mean) / edge_std
        else:
            normalized_graph['edge_features'] = graph['edge_features']  # Empty array
        normalized_graphs.append(normalized_graph)
    
    normalization_stats = {
        'node_mean': node_mean,
        'node_std': node_std,
        'edge_mean': edge_mean,
        'edge_std': edge_std
    }
    
    return normalized_graphs, normalization_stats


def load_dichalcogenides_dataset(
    structures_dir: str = "dichalcogenides_public/structures/",
    targets_file: str = "dichalcogenides_public/targets.csv",
    cutoff_radius: float = 5.0,
    max_samples: int = None,
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[List[Dict], np.ndarray, Dict]:
    """
    Load dichalcogenides dataset and convert structures to graphs.
    
    Args:
        structures_dir: Directory containing structure JSON files
        targets_file: Path to CSV file with targets
        cutoff_radius: Maximum distance (Å) for considering neighbors
        max_samples: Maximum number of samples to load (None for all)
        normalize: Whether to normalize features (recommended)
        verbose: Whether to show progress
        
    Returns:
        Tuple of (graphs, targets, normalization_stats) where:
            - graphs: List of graph dictionaries
            - targets: np.ndarray of shape (num_samples, 1) with band gap values
            - normalization_stats: Dictionary with normalization statistics (None if normalize=False)
    """
    # Load targets
    targets_dict = {}
    with open(targets_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            struct_id = row['_id']
            band_gap = float(row['band_gap'])
            targets_dict[struct_id] = band_gap
    
    # Load structures and convert to graphs
    structure_files = os.listdir(structures_dir)
    if max_samples is not None:
        structure_files = structure_files[:max_samples]
    
    graphs = []
    targets = []
    
    iterator = tqdm.tqdm(structure_files, disable=not verbose) if verbose else structure_files
    
    for filename in iterator:
        struct_id = filename.replace('.json', '')
        
        # Skip if no target available
        if struct_id not in targets_dict:
            continue
        
        # Load structure
        filepath = os.path.join(structures_dir, filename)
        try:
            with open(filepath, 'r') as f:
                d = json.loads(f.read())
                structure = Structure.from_dict(d)
            
            # Convert to graph
            graph = structure_to_graph(structure, cutoff_radius=cutoff_radius)
            graphs.append(graph)
            targets.append(targets_dict[struct_id])
        except Exception as e:
            if verbose:
                print(f"Error loading {filename}: {e}")
            continue
    
    targets = np.array(targets, dtype=np.float64).reshape(-1, 1)
    
    # Normalize features if requested
    normalization_stats = None
    if normalize:
        if verbose:
            print("Normalizing features...")
        graphs, normalization_stats = normalize_features(graphs)
    
    return graphs, targets, normalization_stats


def train_gnn_dichalcogenides(
    structures_dir: str = "dichalcogenides_public/structures/",
    targets_file: str = "dichalcogenides_public/targets.csv",
    cutoff_radius: float = 5.0,
    node_features: int = 5,
    edge_features: int = 3,
    hidden_features: int = 64,
    num_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    train_split: float = 0.8,
    max_samples: int = None,
    normalize: bool = True,
    gradient_clip: float = 1.0,
    optimizer = None,
    verbose: bool = True
) -> Tuple[GNN, Dict]:
    """
    Train GNN model on dichalcogenides dataset.
    
    Args:
        structures_dir: Directory containing structure JSON files
        targets_file: Path to CSV file with targets
        cutoff_radius: Maximum distance (Å) for considering neighbors
        node_features: Number of node features (should match structure_to_graph output)
        edge_features: Number of edge features (should match structure_to_graph output)
        hidden_features: Number of hidden features in GNN layers
        num_layers: Number of GNN layers
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate (used if optimizer is None)
        train_split: Fraction of data to use for training
        max_samples: Maximum number of samples to load (None for all)
        normalize: Whether to normalize features (highly recommended, default True)
        gradient_clip: Maximum gradient norm for clipping (None to disable, default 1.0)
        optimizer: Optional optimizer instance (e.g., Adam, SGD). If None, uses SGD with learning_rate
        verbose: Whether to show progress
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Load dataset
    if verbose:
        print("Loading dataset...")
    graphs, targets, normalization_stats = load_dichalcogenides_dataset(
        structures_dir=structures_dir,
        targets_file=targets_file,
        cutoff_radius=cutoff_radius,
        max_samples=max_samples,
        normalize=normalize,
        verbose=verbose
    )
    
    num_samples = len(graphs)
    if verbose:
        print(f"Loaded {num_samples} samples")
        print(f"Node features shape: {graphs[0]['node_features'].shape}")
        print(f"Edge features shape: {graphs[0]['edge_features'].shape}")
        print(f"Targets shape: {targets.shape}")
        print(f"Target range: [{targets.min():.4f}, {targets.max():.4f}]")
    
    # Split into train and validation
    indices = np.random.permutation(num_samples)
    split_idx = int(train_split * num_samples)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    graphs_train = [graphs[i] for i in train_indices]
    targets_train = targets[train_indices]
    graphs_val = [graphs[i] for i in val_indices]
    targets_val = targets[val_indices]
    
    if verbose:
        print(f"\nTrain samples: {len(graphs_train)}")
        print(f"Validation samples: {len(graphs_val)}")
    
    # Create model
    model = GNN(
        node_features=node_features,
        edge_features=edge_features,
        hidden_features=hidden_features,
        num_classes=1,  # Regression: single output
        num_layers=num_layers
    )
    
    if verbose:
        print(f"\nModel created:")
        print(f"  Node features: {node_features}")
        print(f"  Edge features: {edge_features}")
        print(f"  Hidden features: {hidden_features}")
        print(f"  Number of layers: {num_layers}")
        if optimizer is not None:
            print(f"  Optimizer: {optimizer.__class__.__name__}")
        else:
            print(f"  Optimizer: SGD (learning_rate={learning_rate})")
    
    # Loss function
    loss_fn = MSE()
    
    # Train model
    if verbose:
        print("\nTraining model...")
    history = model.train(
        graphs_train, targets_train,
        graphs_val=graphs_val,
        y_val=targets_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        optimizer=optimizer,
        verbose=verbose,
        gradient_clip=gradient_clip if gradient_clip > 0 else None
    )
    
    if verbose:
        print(f"\nFinal training loss: {history['train_loss'][-1]:.4f}")
        if 'val_loss' in history:
            print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    return model, history


if __name__ == "__main__":
    # Example usage
    model, history = train_gnn_dichalcogenides(
        structures_dir="dichalcogenides_public/structures/",
        targets_file="dichalcogenides_public/targets.csv",
        cutoff_radius=5.0,
        node_features=5,
        edge_features=3,
        hidden_features=64,
        num_layers=3,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        train_split=0.8,
        max_samples=None,  # Use all samples
        verbose=True
    )
    
    print("\nTraining completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if 'val_loss' in history:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
