import torch.nn as nn
import torch


class TinyCNNCombo(nn.Module):
    def __init__(self, num_class):
        super(TinyCNNCombo, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=600)
        self.drop = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=num_class)

        self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU()
        self.outfc1 = nn.Linear(num_class + 10, 16)
        self.outfc2 = nn.Linear(16, num_class)
        
    def forward(self, x, prob):
        prob = self.softmax(prob)

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        out = torch.cat([out, prob], dim=1)
        out = self.relu(self.outfc1(out))
        out = self.outfc2(out)
        
        return out