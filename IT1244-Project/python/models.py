import numpy as np
import torch.nn as nn
import torch
import torch.functional as F
import torch.optim as optim
import librosa

# Baseline MLP
class MLP(nn.Module):
    def __init__(self):
        return None

    def forward(self, x):
        return None

# MLP Transformers

# Baseline CNN
class CNN(nn.Module):
    def __init__(self, n_fft, hop):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(116480, 2)
        self.softmax = nn.Softmax()

    def forward(self, input):
        # print(f"Input shape: {input.shape}")
        x = self.conv1(input)
        # print(f"After conv1: {x.shape}")
        x = self.conv2(x)
        # print(f"After conv2: {x.shape}")
        x = self.conv3(x)
        # print(f"After conv3: {x.shape}")
        x = self.conv4(x)
        # print(f"After conv4: {x.shape}")
        x = self.flatten(x)
        # print(f"After flatten: {x.shape}")
        logits = self.linear(x)
        # print(f"After linear: {x.shape}")
        predictions = self.softmax(logits)
        # print(f"After softmax: {x.shape}")
        return predictions

    
# Transformers CNN

# Transformers CNN Transformers

# CNN Transformers