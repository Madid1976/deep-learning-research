
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple Convolutional Neural Network (CNN) for demonstration
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # Assuming input image size is 32x32, after two pooling layers (2x2 each), it becomes 8x8
        # 64 channels * 8 * 8 = 4096 features
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Apply first conv -> relu -> pool
        x = self.pool1(self.relu1(self.conv1(x)))
        # Apply second conv -> relu -> pool
        x = self.pool2(self.relu2(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        # Apply first fully connected -> relu
        x = self.relu3(self.fc1(x))
        # Apply second fully connected layer for classification
        x = self.fc2(x)
        return x

# Function to train the model (dummy training loop)
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train() # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad() # Zero the parameter gradients
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward() # Backward pass
            optimizer.step() # Optimize weights
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Function to evaluate the model (dummy evaluation)
def evaluate_model(model, test_loader):
    model.eval() # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # Disable gradient calculation during evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")
    return accuracy

# Dummy data loaders for demonstration
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, image_size=32, num_classes=10):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random image (3 channels, image_size x image_size)
        image = torch.randn(3, self.image_size, self.image_size)
        # Generate random label
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label

if __name__ == "__main__":
    # Hyperparameters
    num_classes = 10
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 3

    # Create model, loss function, and optimizer
    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create dummy datasets and data loaders
    train_dataset = DummyDataset(num_samples=500, num_classes=num_classes)
    test_dataset = DummyDataset(num_samples=100, num_classes=num_classes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Starting training...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    print("Training finished.")

    print("Starting evaluation...")
    evaluate_model(model, test_loader)
    print("Evaluation finished.")
