import torch.nn as nn
class ForwardPropertiesNN(nn.Module):
    def __init__(self, input_size=8, output_size=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.network(x)
