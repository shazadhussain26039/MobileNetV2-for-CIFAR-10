import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import MobileNetV2

class MobileNetV2CIFAR(nn.Module):
    def __init__(
        self,
        num_classes=10,
        width_mult=1.0,
        dropout=0.2,
        bn_momentum=0.1,
        bn_eps=1e-5,
        pretrained=False,
    ):
        super().__init__()

        base = MobileNetV2(width_mult=width_mult)

        if pretrained:
            from torchvision.models import mobilenet_v2
            base = mobilenet_v2(weights="DEFAULT")

        # Fix first conv for CIFAR-10 (32x32)
        first_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            3,
            first_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Adjust BN
        for m in base.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps

        self.features = base.features
        last_channel = base.last_channel

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(last_channel, num_classes)

        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def mobilenet_v2_cifar10(**kwargs):
    return MobileNetV2CIFAR(**kwargs)
