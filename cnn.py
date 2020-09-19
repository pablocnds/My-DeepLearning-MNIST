import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)   # 24x24x6
        self.pool1 = nn.MaxPool2d(2, 2)                         # 12x12x6
        self.dp1 = nn.Dropout2d(0.2)
        self.bn1 = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # 8x8x16
        self.pool2 = nn.MaxPool2d(2, 2)                         # 4x4x16
        self.bn2 = nn.BatchNorm2d(16)
        
        # Reshape to (batch_size, -1)
        self.fl1 = nn.Linear(256, 120)                          # 256
        self.fl2 = nn.Linear(120, 10)                           # 84
        # Apply softmax at the end



    def forward(self, x):
        # Preprocess: reshape batch to (batch_size, 1 channel, 28 height, 28 width)
        x = x.view(-1, 1, 28, 28)

        # Convolutional layers
        x = self.bn1(self.dp1(self.pool1(self.conv1(x))))
        x = self.bn2(self.pool2(self.conv2(x))))
        
        # Reshape the feature maps information to a single vector per batch item
        x = x.view(x.size(0), -1)

        # Apply 2 dense layers
        x = F.relu(self.fl1(x))
        x = self.fl2(x)

        # End with softmax
        return F.softmax(x, dim=1)
