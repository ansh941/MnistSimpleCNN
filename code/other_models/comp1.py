import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelComp1(nn.Module):
    def __init__(self):
        super(ModelComp1, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, bias=False, padding=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 5, bias=False, padding=2)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(6272, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)
    def get_logits(self, x):
        x = (x - 0.5) * 2.0
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))        # becomes 28x28
        pool1 = self.pool1(conv1)                           # becomes 14x14
        conv2 = F.relu(self.conv2_bn(self.conv2(pool1)))    # becomes 14x14
        pool2 = self.pool2(conv2)                           # becomes 7x7
        flat1 = torch.flatten(pool2.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat1))
        return logits
    def forward(self, x):
        logits = self.get_logits(x)
        return F.log_softmax(logits, dim=1)
