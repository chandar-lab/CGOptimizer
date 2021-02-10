from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetEncoder(nn.Module):
    """
    Convolutional Encoder for MNIST or similar data. To be used with ClassDecoder in
    conjunction with the EncoderDecoder class
    """

    def __init__(self):
        super(ConvNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        # outputs 9216 dimension vector
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        return x


class ClassDecoder(nn.Module):
    """
     Decoder for MNIST or similar data. To be used with ConvNetEncoder in
     conjunction with the EncoderDecoder class
     """
    def __init__(self):
        super(ClassDecoder, self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class FCLayer(torch.nn.Module):
    """
    Fully-connected MLP with one hidden layer.
    """
    def __init__(self, input_dim=784, hidden_dim=32, output_dim=10, data='image'):
        super(FCLayer, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.data = data

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output, None


class LogisticRegression(torch.nn.Module):
    """
    Simple logistic regression model implemented using a one-layer network
    """
    def __init__(self, input_dim, output_dim, data='image'):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.data = data

    def forward(self, x):
        outputs = self.linear(x)
        return outputs, None
