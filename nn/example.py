import numpy as np
from .mlp import MLP
from .layers.convolution import Conv2d
from .layers.pooling import MaxPool2d, AvgPool2d
from .losses.mse import MSE


def example_mlp():
    """Example of using the MLP"""
    # Create MLP with layers [input_dim=10, hidden_dim=20, output_dim=5]
    model = MLP([10, 20, 5])
    
    # Create sample data
    X = np.random.randn(32, 10)  # 32 samples, 10 features
    y = np.random.randn(32, 5)   # 32 samples, 5 outputs
    
    # Forward pass
    output = model.forward(X)
    print(f"Output shape: {output.shape}")
    
    # Compute loss
    loss_fn = MSE()
    loss = loss_fn.forward(output.T, y.T)  # Transpose for loss function
    print(f"Loss: {np.mean(loss)}")
    
    # Backward pass
    loss_grad = loss_fn.backward(output.T, y.T)
    model.backward(loss_grad.T)  # Transpose back
    
    # Update parameters
    model.update_params(learning_rate=0.001)
    
    # Zero gradients
    model.zero_grad()
    
    print("MLP example completed successfully!")


def example_cnn():
    """Example of using Conv2d and Pooling layers"""
    # Create sample data (batch_size=4, channels=3, height=32, width=32)
    X = np.random.randn(4, 3, 32, 32)
    
    # Create convolutional layer
    conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    
    # Forward pass
    conv_output = conv.forward(X)
    print(f"Conv output shape: {conv_output.shape}")
    
    # Create max pooling layer
    max_pool = MaxPool2d(kernel_size=2, stride=2)
    
    # Forward pass through max pooling
    max_pool_output = max_pool.forward(conv_output)
    print(f"Max pool output shape: {max_pool_output.shape}")
    
    # Create average pooling layer
    avg_pool = AvgPool2d(kernel_size=2, stride=2)
    
    # Forward pass through average pooling
    avg_pool_output = avg_pool.forward(conv_output)
    print(f"Avg pool output shape: {avg_pool_output.shape}")
    
    # Backward pass through average pooling
    avg_pool_grad = np.random.randn(*avg_pool_output.shape)
    avg_pool_input_grad = avg_pool.backward(avg_pool_grad)
    print(f"Avg pool input grad shape: {avg_pool_input_grad.shape}")
    
    # Backward pass through max pooling
    max_pool_grad = np.random.randn(*max_pool_output.shape)
    max_pool_input_grad = max_pool.backward(max_pool_grad)
    print(f"Max pool input grad shape: {max_pool_input_grad.shape}")
    
    # Backward pass through convolution
    conv_grad = np.random.randn(*conv_output.shape)
    conv_input_grad = conv.backward(conv_grad)
    print(f"Conv input grad shape: {conv_input_grad.shape}")
    
    # Update convolution parameters
    conv.update_params(learning_rate=0.001)
    
    print("CNN example completed successfully!")


def example_lenet():
    """Example of using the LeNet model"""
    from .cnn import LeNet
    
    # Create sample data (batch_size=4, channels=1, height=28, width=28)
    # This is the typical format for MNIST digits
    X = np.random.randn(4, 1, 28, 28)
    
    # Create LeNet model for 10 classes (digits 0-9)
    model = LeNet(num_classes=10)
    
    # Forward pass
    output = model.forward(X)
    print(f"LeNet output shape: {output.shape}")  # Should be (4, 10)
    
    # Create sample target (one-hot encoded labels)
    y_true = np.random.randn(4, 10)
    
    # Compute loss (using MSE as an example)
    loss_fn = MSE()
    loss = loss_fn.forward(output.T, y_true.T)  # Transpose for loss function
    print(f"Loss: {np.mean(loss)}")
    
    # Backward pass
    loss_grad = loss_fn.backward(output.T, y_true.T)
    model.backward(loss_grad.T)  # Transpose back
    
    # Update parameters
    model.update_params(learning_rate=0.001)
    
    # Zero gradients
    model.zero_grad()
    
    print("LeNet example completed successfully!")
    

def example_lenet_training():
    """Example of training the LeNet model with cross entropy loss"""
    from .cnn import LeNet
    from .losses.cross_entropy import CrossEntropy
    
    # Create sample training data (batch_size=100, channels=1, height=28, width=28)
    X_train = np.random.randn(100, 1, 28, 28)
    # Create one-hot encoded labels for 10 classes
    y_train = np.eye(10)[np.random.choice(10, 100)].astype(np.float32)
    
    # Create sample validation data
    X_val = np.random.randn(20, 1, 28, 28)
    y_val = np.eye(10)[np.random.choice(10, 20)].astype(np.float32)
    
    # Create LeNet model for 10 classes
    model = LeNet(num_classes=10)
    
    # Create CrossEntropy loss function
    loss_fn = CrossEntropy()
    
    # Train the model
    print("Training LeNet with CrossEntropy loss...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        batch_size=16,
        learning_rate=0.001,
        loss_fn=loss_fn,
        verbose=True
    )
    
    print("Training completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # Test forward pass after training
    test_output = model.forward(X_val[:5])
    print(f"Test output shape: {test_output.shape}")
    print("LeNet training example completed successfully!")


def example_transformer():
    """Example of using the Transformer model for sequence classification"""
    from .transformer_model import Transformer
    from .losses.cross_entropy import CrossEntropy
    
    # Model hyperparameters
    n_inputs = 64      # Input feature dimension (e.g., embedding size)
    d_model = 128     # Model dimension (must be divisible by n_heads)
    n_heads = 8       # Number of attention heads
    n_layers = 2      # Number of transformer blocks
    d_ff = 512        # Feed-forward dimension
    n_outputs = 10    # Number of output classes
    seq_len = 20      # Sequence length
    
    # Create transformer model
    model = Transformer(
        n_inputs=n_inputs,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        n_outputs=n_outputs,
        max_seq_len=seq_len
    )
    
    # Create sample sequence data: (seq_len, n_inputs)
    X = np.random.randn(seq_len, n_inputs)
    
    # Forward pass
    output = model.forward(X)
    print(f"Transformer output shape: {output.shape}")  # Should be (n_outputs,)
    print(f"Output logits: {output[:5]}")  # Print first 5 logits
    
    # Create sample target (one-hot encoded label)
    y_true = np.eye(n_outputs)[0]  # Class 0
    
    # Compute loss
    loss_fn = CrossEntropy()
    loss = loss_fn.forward(output.T, y_true.T)
    print(f"Loss: {np.mean(loss):.4f}")
    
    # Backward pass
    loss_grad = loss_fn.backward(output.T, y_true.T)
    model.backward(loss_grad.T)
    
    # Update parameters
    model.update_params(learning_rate=0.001)
    
    # Zero gradients
    model.zero_grad()
    
    print("Transformer basic example completed successfully!")


def example_transformer_training():
    """Example of training the Transformer model for sequence classification"""
    from .transformer_model import Transformer
    from .losses.cross_entropy import CrossEntropy
    
    # Model hyperparameters
    n_inputs = 64      # Input feature dimension
    d_model = 128     # Model dimension
    n_heads = 8        # Number of attention heads
    n_layers = 2       # Number of transformer blocks
    d_ff = 512         # Feed-forward dimension
    n_outputs = 10    # Number of output classes
    seq_len = 20       # Sequence length
    num_samples = 200 # Number of training samples
    
    # Create transformer model
    model = Transformer(
        n_inputs=n_inputs,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        n_outputs=n_outputs,
        max_seq_len=seq_len
    )
    
    # Create sample training data
    # Shape: (num_samples, seq_len, n_inputs)
    X_train = np.random.randn(num_samples, seq_len, n_inputs)
    # Create one-hot encoded labels for n_outputs classes
    y_train = np.eye(n_outputs)[np.random.choice(n_outputs, num_samples)].astype(np.float32)
    
    # Create sample validation data
    X_val = np.random.randn(40, seq_len, n_inputs)
    y_val = np.eye(n_outputs)[np.random.choice(n_outputs, 40)].astype(np.float32)
    
    # Create CrossEntropy loss function
    loss_fn = CrossEntropy()
    
    # Train the model
    print("Training Transformer with CrossEntropy loss...")
    print(f"Model architecture: {n_layers} layers, {n_heads} heads, d_model={d_model}, d_ff={d_ff}")
    print(f"Training on {num_samples} sequences of length {seq_len}")
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        batch_size=16,
        learning_rate=0.001,
        loss_fn=loss_fn,
        verbose=True
    )
    
    print("\nTraining completed!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        if history['val_accuracy']:
            print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # Test forward pass after training
    test_output = model.forward(X_val[0])  # Single sequence
    print(f"\nTest output shape: {test_output.shape}")
    predicted_class = np.argmax(test_output)
    true_class = np.argmax(y_val[0])
    print(f"Predicted class: {predicted_class}, True class: {true_class}")
    print("Transformer training example completed successfully!")


if __name__ == "__main__":
    # example_mlp()
    # print("\n" + "="*50 + "\n")
    example_cnn()
    print("\n" + "="*50 + "\n")
    example_lenet()
    print("\n" + "="*50 + "\n")
    # example_lenet_training()
    # print("\n" + "="*50 + "\n")
    example_transformer()
    print("\n" + "="*50 + "\n")
    example_transformer_training()
