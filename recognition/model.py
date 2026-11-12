"""Checkpoint-compatible architecture extracted from notebook Part 3."""
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.activation(self.main(inputs) + self.skip(inputs))

class SevenHeadRecognizer(nn.Module):
    def __init__(self, class_counts):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d((1, 12)),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 12, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
        )
        self.heads = nn.ModuleList(nn.Linear(512, count) for count in class_counts)

    def forward(self, inputs):
        features = self.shared(self.features(inputs))
        return [head(features) for head in self.heads]
