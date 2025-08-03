import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),  # layers.0
            nn.ReLU(),                    # layers.1
            nn.Linear(1024, 128),         # layers.2
            nn.ReLU(),                    # layers.3
            nn.Linear(128, 64),           # layers.4
            nn.ReLU(),                    # layers.5
            nn.Linear(64, 16),            # layers.6
            # sem ReLU aqui
            nn.Linear(16, 1)              # layers.7
        )

    def forward(self, x):
        return self.layers(x)
