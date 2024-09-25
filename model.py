import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F

class QuantizedCNN(nn.Module):
    def __init__(self):
        super(QuantizedCNN, self).__init__()
        # Define the first quantized convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Define the first max pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define the second quantized convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Define the second max pooling layer
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)
        
        # Define the quantized dense layer
        self.fc1 = nn.Linear(16 * 5 * 5, 10)  # Adjust the sizing according to your model's final pooling output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.softmax(x)
        
        return x

# Create the model instance
model = QuantizedCNN().to('cuda')



# Print the model summary

summary(model, input_size=(1, 28, 28))

