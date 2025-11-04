import numpy as np
from .mlp import MLP
from .layers.convolution import Conv2d
from .layers.pooling import MaxPool2d
from .layers.linear import Linear
from .layers.sigmoid import Sigmoid
from .losses.mse import MSE


def test_mlp():
    """Test MLP functionality"""
    print("Testing MLP...")
    
    # Create MLP: 4 -> 8 -> 3
    model = MLP([4, 8, 3])
    
    # Test forward pass
    X = np.random.randn(10, 4)  # 10 samples, 4 features
    output = model.forward(X)
    assert output.shape == (10, 3), f"Expected (10, 3), got {output.shape}"
    
    # Test backward pass
    loss_fn = MSE()
    y_true = np.random.randn(10, 3)
    loss = loss_fn.forward(output.T, y_true.T)
    loss_grad = loss_fn.backward(output.T, y_true.T)
    model.backward(loss_grad)
    
    # Test parameter update
    model.update_params(0.001)
    
    # Test zero grad
    model.zero_grad()
    
    print("MLP test passed!")


def test_conv2d():
    """Test Conv2d functionality"""
    print("Testing Conv2d...")
    
    # Create conv layer
    conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    
    # Test forward pass
    X = np.random.randn(2, 3, 8, 8)  # 2 samples, 3 channels, 8x8
    output = conv.forward(X)
    assert output.shape == (2, 16, 8, 8), f"Expected (2, 16, 8, 8), got {output.shape}"
    
    # Test backward pass
    grad_output = np.random.randn(2, 16, 8, 8)
    grad_input = conv.backward(grad_output)
    assert grad_input.shape == X.shape, f"Expected {X.shape}, got {grad_input.shape}"
    
    # Test parameter update
    conv.update_params(0.001)
    
    print("Conv2d test passed!")


def test_maxpool2d():
    """Test MaxPool2d functionality"""
    print("Testing MaxPool2d...")
    
    # Create max pool layer
    pool = MaxPool2d(kernel_size=2, stride=2)
    
    # Test forward pass
    X = np.random.randn(2, 3, 8, 8)  # 2 samples, 3 channels, 8x8
    output = pool.forward(X)
    assert output.shape == (2, 3, 4, 4), f"Expected (2, 3, 4, 4), got {output.shape}"
    
    # Test backward pass
    grad_output = np.random.randn(2, 3, 4, 4)
    grad_input = pool.backward(grad_output)
    assert grad_input.shape == X.shape, f"Expected {X.shape}, got {grad_input.shape}"
    
    print("MaxPool2d test passed!")


def test_linear():
    """Test Linear layer functionality"""
    print("Testing Linear layer...")
    
    # Create linear layer
    linear = Linear(10, 5)
    
    # Test forward pass
    X = np.random.randn(20, 10)  # 20 samples, 10 features
    output = linear.forward(X.T)  # Transpose for layer format
    assert output.shape == (5, 20), f"Expected (5, 20), got {output.shape}"

    # Test backward pass
    grad_output = np.random.randn(5, 20)
    grad_input = linear.backward(grad_output)
    assert grad_input.shape == X.T.shape, f"Expected {X.T.shape}, got {grad_input.shape}"
    
    # Test parameter update
    linear.update_params(0.001)
    
    print("Linear layer test passed!")


def test_sigmoid():
    """Test Sigmoid layer functionality"""
    print("Testing Sigmoid layer...")
    
    # Create sigmoid layer
    sigmoid = Sigmoid()
    
    # Test forward pass
    X = np.random.randn(5, 10)  # 10 samples, 5 features
    output = sigmoid.forward(X)
    assert output.shape == X.shape, f"Expected {X.shape}, got {output.shape}"
    # Check that output is in (0, 1) range
    assert np.all(output > 0) and np.all(output < 1), "Sigmoid output should be in (0, 1) range"
    
    # Test backward pass
    grad_output = np.random.randn(5, 10)
    grad_input = sigmoid.backward(grad_output)
    assert grad_input.shape == X.shape, f"Expected {X.shape}, got {grad_input.shape}"
    
    print("Sigmoid layer test passed!")


def test_mse():
    """Test MSE loss functionality"""
    print("Testing MSE loss...")
    
    # Create MSE loss
    mse = MSE()
    
    # Test forward pass
    y_pred = np.random.randn(3, 10)  # 3 outputs, 10 samples
    y_true = np.random.randn(3, 10)  # 3 outputs, 10 samples
    loss = mse.forward(y_pred, y_true)
    assert loss.shape == (10,), f"Expected (10,), got {loss.shape}"
    
    # Test backward pass
    loss_grad = mse.backward(y_pred, y_true)
    assert loss_grad.shape == y_pred.shape, f"Expected {y_pred.shape}, got {loss_grad.shape}"
    
    print("MSE loss test passed!")
    

def test_cross_entropy():
    """Test CrossEntropy loss functionality"""
    print("Testing CrossEntropy loss...")
    
    # Import inside function to avoid circular imports
    from .losses.cross_entropy import CrossEntropy
    
    # Create CrossEntropy loss
    ce = CrossEntropy()
    
    # Test forward pass
    # Create predictions (logits)
    y_pred = np.array([[2.0, 1.0, 0.1],
                       [1.0, 2.0, 0.1],
                       [0.1, 1.0, 2.0]])  # 3 classes, 3 samples
    
    # Create one-hot encoded true labels
    y_true = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]])  # 3 classes, 3 samples
    
    loss = ce.forward(y_pred, y_true)
    assert loss.shape == (3,), f"Expected (3,), got {loss.shape}"
    
    # Test backward pass
    loss_grad = ce.backward(y_pred, y_true)
    assert loss_grad.shape == y_pred.shape, f"Expected {y_pred.shape}, got {loss_grad.shape}"
    
    print("CrossEntropy loss test passed!")

def test_lenet():
    """Test LeNet functionality"""
    print("Testing LeNet...")
    
    # Import inside function to avoid circular imports
    from .cnn import LeNet
    
    # Create LeNet model
    model = LeNet(num_classes=10)
    
    # Test forward pass
    X = np.random.randn(2, 1, 28, 28)  # 2 samples, 1 channel, 28x28
    output = model.forward(X)
    assert output.shape == (2, 10), f"Expected (2, 10), got {output.shape}"
    
    # Test backward pass
    loss_grad = np.random.randn(2, 10)
    model.backward(loss_grad)
    
    # Test parameter update
    model.update_params(0.001)
    
    # Test zero grad
    model.zero_grad()
    
    print("LeNet test passed!")


def run_all_tests():
    """Run all tests"""
    print("Running all tests...\n")
    
    test_linear()
    test_sigmoid()
    test_mse()
    test_conv2d()
    test_maxpool2d()
    test_mlp()
    test_lenet()
    test_cross_entropy()
    
    print("\nAll tests passed! Framework is working correctly.")


if __name__ == "__main__":
    run_all_tests()