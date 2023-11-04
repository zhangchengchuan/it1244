import numpy as np
import torch.nn as nn
import torch
import torch.functional as F
import torch.optim as optim
import torch.nn.init as init


# Xavier initialization for different layers
def init_weights(layer):
    if isinstance(layer, nn.Conv2d):
        init.xavier_normal_(layer.weight)


# Baseline Multi-layer Perceptron (MLP)
class MLP(nn.Module):
    """Baseline MLP model. We use LeakyReLU activation function and dropout to prevent over-fitting."""

    def __init__(self, dropout=0.2):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(200900, 2048),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        self.fc1.apply(init_weights)

        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        self.fc2.apply(init_weights)

        self.fc3 = nn.Sequential(
            nn.Linear(1024, 2),
        )
        self.fc3.apply(init_weights)

    """Forward pass"""

    def forward(self, x):
        # We flatten the input to a vector before passing it through the fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.fc3(x)
        return logits


# Baseline CNN
class CNN(nn.Module):
    """Convolutional Neural Network CNN model. We use LeakyReLU activation function and dropout to
    prevent over-fitting."""

    def __init__(self, dropout=0.2):
        super(CNN, self).__init__()

        """
        Convolutional layers. We use 4 convolutional layers with max pooling and batch normalization. As we have
        learnt in class, batch normalization helps to speed up training. It also helps to make the training process
        more stable and less likely to get stuck in a local minima. Max pooling helps to reduce the dimensionality of
        the input, which helps to reduce the number of parameters in the model. This helps to reduce the training time.
        
        Most importantly, it makes the CNN more robust to small changes in the input. For example, if we have a picture
        of a cat, we can still recognize it as a cat even if the cat is not in the center of the picture, or is rotated 
        in another direction. This is because max pooling helps to reduce the dimensionality of the input, which makes 
        the CNN more robust to small changes in the input.
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv1.apply(init_weights)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2.apply(init_weights)

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3.apply(init_weights)

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4.apply(init_weights)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(116480, 2)

    """Forward pass"""

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        return logits


class LSTM(nn.Module):
    """
    LSTM stands for Long Short-Term Memory. It is a type of recurrent neural network (RNN). RNNs are used for sequential
    data, such as time series data. In our case, we use it for audio data. The input to the LSTM is a sequence of audio
    frames. The output is a sequence of hidden states. The hidden states are then passed to a fully connected layer to
    produce the final output.

    What is good about this model is that it is better than traditional RNNs at capturing long-term dependencies.
    Traditional RNNs have difficulty learning long-term dependencies in sequential data. This is because the
    gradients of RNNs tend to vanish or explode over time, making it difficult to train the network to learn long-term
    patterns.
    """

    def __init__(self, dropout=0.2):
        super(LSTM, self).__init__()

        self.flatten1 = nn.Flatten()
        self.lstm = nn.LSTM(
            input_size=1025,
            hidden_size=256,
            batch_first=True,
            num_layers=2,
            dropout=dropout,
            bidirectional=True
        )
        self.lstm.apply(init_weights)
        self.fc = nn.Linear(2 * 256, 2)

    def forward(self, x):
        # Expecting input of shape (batch, seq_len, features), where
        # seq_len is the number of time steps and features is the number of frequency bins

        # Reshape input to (batch_size, time_steps, frequency_bins)
        x = x.permute(0, 3, 1, 2).reshape(x.size(0), x.size(3), -1)

        # Forward propagate LSTM
        x, (h_n, c_n) = self.lstm(x)

        # You might need to extract just the final hidden state from the last time step
        # If your LSTM is bidirectional, it concatenates the final forwards (h_n[-2,:,:])
        # and backwards (h_n[-1,:,:]) hidden states
        x = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)

        # Pass the final state through the fully connected layer
        logits = self.fc(x)

        return logits
