import numpy as np
from .cross_entropy import CrossEntropy


def test_cross_entropy():
    """Test CrossEntropy loss functionality"""
    print("Testing CrossEntropy loss...")
    
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
    
    # For correct predictions, gradient should be small
    # For incorrect predictions, gradient should be larger
    print(f"Loss values: {loss}")
    print(f"Gradient values: {loss_grad}")
    
    print("CrossEntropy loss test passed!")


if __name__ == "__main__":
    test_cross_entropy()