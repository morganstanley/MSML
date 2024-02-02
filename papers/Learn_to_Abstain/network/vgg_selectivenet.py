import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Union, List, Dict, Any, cast


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

model_path = {
    'vgg16': '/v/campus/vi/appl/msml/dev/data/footprint-00/icml2023slearnlogdir/checkpoint/vgg16-397923af.pth'
}

class VGGSelective(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        num_class: int = 1000,
        init_weights: bool = True, 
        **kwargs
    ) -> None:

        super(VGGSelective, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # extra linear projection 
        self.fc = nn.Linear(in_features=512*7*7, out_features=512)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(512)
        self.drop = nn.Dropout(0.5)
        # classification head (f)
        self.cls_fc = nn.Linear(in_features=512, out_features=num_class)
        # selection head (g)
        self.sel_fc1 = nn.Linear(in_features=512, out_features=512)
        self.sel_bn = nn.BatchNorm1d(512)
        self.sel_fc2 = nn.Linear(in_features=512, out_features=1)
        self.sigmoid = nn.Sigmoid()
        # auxiliary head (h)
        self.aux_fc = nn.Linear(in_features=512, out_features=num_class)

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(self.bn(self.relu(self.fc(x)))) # special procedure from https://github.com/geifmany/selectivenet/blob/a6d0a8fd33dae61da910b61a2aae93102d2d4869/models/svhn_vgg_selectivenet.py#L129 
        
        # classification head
        cls_out = self.cls_fc(x)
        # selection head
        sel_out = self.sel_fc1(x)
        sel_out = self.sel_bn(sel_out)
        sel_out = self.sel_fc2(sel_out)
        sel_out = self.sigmoid(sel_out)
        # auxiliary head
        aux_out = self.aux_fc(x)
        
        return cls_out, sel_out, aux_out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, num_class: int, **kwargs: Any) -> VGGSelective:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGGSelective(make_layers(cfgs[cfg], batch_norm=batch_norm), num_class=num_class, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        state_dict = torch.load(model_path[arch])
        model_dict.update({k:v for k, v in state_dict.items() if k in model_dict})
        model.load_state_dict(model_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGGSelective:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)



def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGGSelective:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)



def vgg13(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGGSelective:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)



def vgg13_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGGSelective:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)



def vgg16selectivenet(num_class: int, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGGSelective:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, num_class=num_class, **kwargs)



def vgg16_bnselectivenet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGGSelective:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress,  num_class=num_class, **kwargs)



def vgg19(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGGSelective:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)



def vgg19_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGGSelective:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)