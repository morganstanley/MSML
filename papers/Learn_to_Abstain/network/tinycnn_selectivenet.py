import torch.nn as nn
import torch


class TinyCNNSelectiveNet(nn.Module):
    def __init__(self, num_class, *args, **kwargs):
        super(TinyCNNSelectiveNet, self).__init__()
        
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
        # extra linear projection
        self.fc = nn.Linear(in_features=64*7*7, out_features=512)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(0.5)
        # classification head (f)
        self.cls_fc = nn.Linear(in_features=512, out_features=num_class)
        # selectoin head (g)
        self.sel_fc1 = nn.Linear(in_features=512, out_features=512)
        self.sel_bn = nn.BatchNorm1d(512)
        self.sel_fc2 = nn.Linear(in_features=512, out_features=1)
        # auxiliary head (h)
        self.aux_fc1 = nn.Linear(in_features=512, out_features=num_class)
        
    def forward(self, x, *args, **kwargs):
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        # special procedure from https://github.com/geifmany/selectivenet/blob/a6d0a8fd33dae61da910b61a2aae93102d2d4869/models/svhn_vgg_selectivenet.py#L129 
        out = self.drop(self.bn(self.relu(self.fc(out)))) 
        # classification head
        cls_out = self.cls_fc(out)
        # selection head
        sel_out = self.sel_fc1(out)
        sel_out = self.sel_bn(sel_out)
        sel_out = self.sel_fc2(sel_out)
        sel_out = torch.sigmoid(sel_out)
        # auxiliary head
        aux_out = self.aux_fc1(out)
        
        return cls_out, sel_out, aux_out
