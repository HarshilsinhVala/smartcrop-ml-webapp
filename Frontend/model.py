import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=42):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling reduces spatial size
        
        # Calculate the feature size dynamically
        self._to_linear = None
        self.calculate_feature_size()

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def calculate_feature_size(self):
        # Create a dummy tensor with the same shape as an input image
        with torch.no_grad():
            x = torch.zeros(1, 3, 128, 128)  # (batch_size=1, channels=3, height=128, width=128)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            self._to_linear = x.numel()  # Get the total number of features after flattening

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)  # Flatten before FC layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
