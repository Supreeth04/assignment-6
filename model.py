import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super(Net, self).__init__()
        
        # First block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second block
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 24, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third block with GAP
        self.conv5 = nn.Conv2d(24, 24, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(24, 16, 1)  # 1x1 conv to reduce channels
        self.bn6 = nn.BatchNorm2d(16)
        
        # Global Average Pooling and final conv
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv7 = nn.Conv2d(16, 10, 1)  # Final 1x1 conv for classification

    def forward(self, x):
        # First block
        x = self.dropout1(self.pool1(
            F.relu(self.bn2(self.conv2(
            F.relu(self.bn1(self.conv1(x)))
        )))))
        
        # Second block
        x = self.dropout2(self.pool2(
            F.relu(self.bn4(self.conv4(
            F.relu(self.bn3(self.conv3(x)))
        )))))
        
        # Third block
        x = F.relu(self.bn6(self.conv6(
            F.relu(self.bn5(self.conv5(x)))
        )))
        
        # GAP and final conv
        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        
        return F.log_softmax(x, dim=1)