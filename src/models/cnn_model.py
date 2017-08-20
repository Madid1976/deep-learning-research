import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate input features for the first fully connected layer
        # Assuming input image size is 28x28 (e.g., MNIST)
        # After conv1, pool1: 28/2 = 14
        # After conv2, pool2: 14/2 = 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten the tensor
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Example usage with dummy data
    input_tensor = torch.randn(1, 1, 28, 28) # Batch size 1, 1 channel, 28x28 image
    model = SimpleCNN(num_classes=10)
    output = model(input_tensor)
    print("Model output shape:", output.shape)
    assert output.shape == (1, 10), "Output shape mismatch!"
    print("SimpleCNN model created and tested successfully!")

    # Test with a larger batch size
    input_tensor_batch = torch.randn(64, 1, 28, 28)
    output_batch = model(input_tensor_batch)
    print("Model output shape with batch:", output_batch.shape)
    assert output_batch.shape == (64, 10), "Output shape mismatch for batch!"
    print("Batch processing successful!")

    # Check number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    assert num_params > 10000, "Model seems too small!"
