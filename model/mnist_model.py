import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)  # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)  # 13x13 -> 11x11
        self.fc1 = nn.Linear(8 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 8 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 