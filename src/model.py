from torch.nn import Module
import torch.nn as nn
import torch
import numpy as np
from torchsummary import summary
from collections import deque


# Stability detector for steady hand detection
class StabilityDetector:
    def __init__(self, max_frames=30, stability_threshold=0.02):
        self.max_frames = max_frames
        self.stability_threshold = stability_threshold
        self.keypoints_history = deque(maxlen=max_frames)
    
    def add_keypoints(self, keypoints):
        self.keypoints_history.append(keypoints)
    
    def is_stable(self):
        if len(self.keypoints_history) < self.max_frames:
            return False
        arr = np.array(self.keypoints_history)
        var = np.var(arr, axis=0)
        return np.mean(var) < self.stability_threshold
    
    def reset(self):
        self.keypoints_history.clear()

class MyCNNs(Module):
    def __init__(self, num_classes=26):
        super(MyCNNs, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 218),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(218, num_classes)
        )

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output.view(output.shape[0], -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


if __name__ == '__main__':
    model = MyCNNs()
    summary(model, (3, 224, 224))

