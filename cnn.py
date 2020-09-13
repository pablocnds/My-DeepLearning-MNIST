import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CNN(nn.Module):

    def __init__(self, h, w, outputs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=2)   # 24x24x6
        self.pool1 = nn.AvgPool2d(2, 1)                         # 12x12x6
        self.bn1 = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=2)  # 8x8x16
        self.pool2 = nn.AvgPool2d(2, 1)                         # 4x4x16
        self.bn2 = nn.BatchNorm2d(16)
        
        # Reshape to 1d
        self.fl1 = nn.Linear(256, 120)                          # 256
        self.fl2 = nn.Linear(120, 84)                           # 84
        self.fl3 = nn.Linear(84, 10)                            # 10
        # Apply softmax at the end



    def forward(self, x):
        # Preprocess: reshape batch to (batch_size, 1 channel, 28 height, 28 width)
        x = x.view(-1, 1, 28, 28)

        # Convolutional layers
        x = self.bn1(self.pool1(self.conv1(x)))
        x = self.bn2(self.pool2(self.conv2(x)))

        # Reshape the feature maps information to a single vector
        x = x.view(x.size(), -1)

        # Apply 3 linear layers
        x = self.fl3(self.fl2(self.fl1(x)))

        # End with softmax
        return F.softmax(x)
