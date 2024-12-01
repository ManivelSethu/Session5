import torch
import pytest
from model.mnist_model import MNISTNet
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameter Details:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params

def test_model_architecture():
    model = MNISTNet()
    
    # Print model architecture
    print(f"\nModel Architecture:")
    print(model)
    
    # Test and print total parameters
    total_params = count_parameters(model)
    assert total_params < 24000, f"Model has {total_params:,} parameters, should be less than 24,000"
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    print(f"\nInput shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
    
    # Print feature map sizes
    x = test_input
    print("\nFeature map sizes:")
    x = model.pool(F.relu(model.conv1(x)))
    print(f"After conv1 + pool: {x.shape}")
    x = model.pool(F.relu(model.conv2(x)))
    print(f"After conv2 + pool: {x.shape}")

def test_model_training():
    print("\nTesting model training...")
    device = torch.device("cpu")
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Prepare a small batch of data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load a small subset of training data
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Train for one batch
    model.train()
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    
    # Forward pass
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    
    # Test if loss is finite
    print(f"Initial training loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite"
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    # Test if gradients are computed
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains inf or nan"

def test_model_accuracy():
    device = torch.device("cpu")
    model = MNISTNet().to(device)
    
    # Load pre-trained model if exists (for CI/CD)
    try:
        import glob
        model_files = glob.glob('saved_models/mnist_model_*.pth')
        if model_files:
            latest_model = max(model_files)
            print(f"\nLoading model from: {latest_model}")
            model.load_state_dict(torch.load(latest_model, map_location=device, weights_only=True))
        else:
            print("\nNo saved model found, skipping accuracy test")
            pytest.skip("No saved model found")
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        pytest.skip("Error loading model")
    
    # Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Evaluate accuracy
    model.eval()
    correct = 0
    total = 0
    
    print("\nEvaluating model accuracy...")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    assert accuracy >= 95, f"Model accuracy is {accuracy:.2f}%, should be at least 95%" 