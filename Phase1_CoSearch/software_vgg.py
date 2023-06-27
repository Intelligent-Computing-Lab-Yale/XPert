'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'Custom': [512, 'M', 512, 256, 256, 'M', 64, 512, 128, 'M', 512, 64, 128,  64, 64],
    'Custom_3': [512, 'M', 512, 512, 32, 'M', 256, 16, 512, 'M', 128, 64, 64,  16, 256],
    'SNN_1': [64, 64, 'M', 64, 512, 'M', 128, 256, 64, 'M', 64, 128, 512, 'M', 256, 64, 512],
    'SNN_2': [64, 128, 'M', 64, 64, 'M', 512, 128, 64, 'M', 64, 256, 64, 'M', 256, 64, 512],
    'SNN_3': [128, 128, 'M', 64, 64, 'M', 128, 256, 128, 'M', 512, 256, 64, 'M', 512, 128, 256],

    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_soft(nn.Module):
    def __init__(self, vgg_name):
        super(VGG_soft, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        out = self.features(x)
        # print(out.size())
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=2, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
