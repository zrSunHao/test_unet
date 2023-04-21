import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, features):
        super(Block, self).__init__()
        self.features = features
        self.conv1 = nn.Conv2d(in_channels = in_channels,
                               out_channels = features,
                               kernel_size = 3,
                               padding = 'same')
        self.BN1 = nn.BatchNorm2d(num_features=self.features)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels = features,
                               out_channels = features,
                               kernel_size = 3,
                               padding = 'same')
        self.BN2 = nn.BatchNorm2d(num_features=self.features)
        self.relu2 = nn.ReLU(inplace=True)
        
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.BN1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.BN2(x)
        x = self.relu2(x)

        return x
