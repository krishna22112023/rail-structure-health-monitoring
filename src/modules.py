import torch
import torch.nn as nn
import torch.nn.functional as F

# SimpleANN remains the same
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_layers=3, dropout=0.2):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()

        layers = []
        # First hidden layer uses LazyLinear to infer input dimension automatically.
        layers.append(nn.LazyLinear(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Repeat additional hidden layers if num_layers > 1
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Final output layer (doesn't include activation or dropout)
        layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.network(x)
        return x

# CNN uses 3 convolutional layers with ReLU and max pooling, then flattens and calls ANN.
class CNN(nn.Module):
    def __init__(self, input_channels, num_cnn_layers, hidden_dim, num_classes,num_layers,dropout,init_channels=32):
        """
        Args:
            input_channels (int): Number of channels in the input image.
            num_cnn_layers (int): Number of convolutional layers.
            hidden_dim (int): Hidden layer size for ANN.
            num_classes (int): Number of output classes.
            init_channels (int): Number of filters in the first CNN layer (doubles each layer).
        """
        super(CNN, self).__init__()
        
        layers = []
        in_channels = input_channels
        out_channels = init_channels
        
        for _ in range(num_cnn_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels *= 2  # Double the filters at each layer
        
        self.cnn = nn.Sequential(*layers)
        self.ann = ANN(hidden_dim, num_classes, num_layers, dropout)
    
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.ann(x)
        return x