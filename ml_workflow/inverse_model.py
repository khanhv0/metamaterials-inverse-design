import torch
import torch.nn as nn
import torch.nn.functional as F

class InverseDesignNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims=[256, 512, 512, 256], dropout=0.2):
        super(InverseDesignNN, self).__init__()

        # Create fully connected layers dynamically
        layers = []
        in_dim = input_size
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))  # Normalize activations
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(dropout))    # Regularization
            in_dim = h_dim

        # Final output layer
        #layers.append(nn.Tanh())  # Use Tanh activation
        layers.append(nn.Linear(in_dim, output_size))

        # Combine all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)