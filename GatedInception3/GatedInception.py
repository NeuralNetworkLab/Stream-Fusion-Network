from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
#import pdb
import numpy as np
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

'''
__all__ = ['Inception3', 'inception_v3']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    # 'inception_v3_google': 'https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth',
}


def inception_v3(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        model = Inception3(**kwargs)
        model.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        return model

    return Inception3(**kwargs)


class Inception3(nn.Module):

    def __init__(self, num_classes=101, aux_logits=False, transform_input=False):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
'''


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight.data, 1.0, 0.02)
                init.constant(m.bias.data, 0.0)

    def forward(self, x):
        x = self.conv(x)
        # print('basic',x.size())
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, nc_in, nc_1x1, nc_3x3_reduce, nc_3x3, nc_double_3x3_reduce, nc_double_3x3_a, nc_double_3x3_b,
                 nc_pool_conv):
        super(Inception, self).__init__()

        self.inception_1x1 = BasicConv2d(nc_in, nc_1x1, kernel_size=1, stride=1)

        self.inception_3x3_reduce = BasicConv2d(nc_in, nc_3x3_reduce, kernel_size=1)
        self.inception_3x3 = BasicConv2d(nc_3x3_reduce, nc_3x3, kernel_size=3, stride=1, padding=1)

        self.inception_double_3x3_reduce = BasicConv2d(nc_in, nc_double_3x3_reduce, kernel_size=1, stride=1)
        self.inception_double_3x3_a = BasicConv2d(nc_double_3x3_reduce, nc_double_3x3_a, kernel_size=3, stride=1,
                                                  padding=1)
        self.inception_double_3x3_b = BasicConv2d(nc_double_3x3_a, nc_double_3x3_b, kernel_size=3, stride=1, padding=1)

        # self.inception_pool = nn.AvgPool2d(kernel_size = 3, stride=1, padding=1)
        self.inception_pool_conv = BasicConv2d(nc_in, nc_pool_conv, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.inception_1x1(x)
        # print('inception',x1.size())
        x2 = self.inception_3x3_reduce(x)
        x2 = self.inception_3x3(x2)
        x3 = self.inception_double_3x3_reduce(x)
        x3 = self.inception_double_3x3_a(x3)
        x3 = self.inception_double_3x3_b(x3)
        x4 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x4 = self.inception_pool_conv(x4)
        out = [x1, x2, x3, x4]
        return torch.cat(out, 1)


class Inception_downsample(nn.Module):
    def __init__(self, nc_in, nc_3x3_reduce, nc_3x3, nc_double_3x3_reduce, nc_double_3x3_a, nc_double_3x3_b):
        super(Inception_downsample, self).__init__()

        self.inception_3x3_reduce = BasicConv2d(nc_in, nc_3x3_reduce, kernel_size=1)
        self.inception_3x3 = BasicConv2d(nc_3x3_reduce, nc_3x3, kernel_size=3, stride=2, padding=1)

        self.inception_double_3x3_reduce = BasicConv2d(nc_in, nc_double_3x3_reduce, kernel_size=1, stride=1)
        self.inception_double_3x3_a = BasicConv2d(nc_double_3x3_reduce, nc_double_3x3_a, kernel_size=3, stride=1,
                                                  padding=1)
        self.inception_double_3x3_b = BasicConv2d(nc_double_3x3_a, nc_double_3x3_b, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x2 = self.inception_3x3_reduce(x)
        x2 = self.inception_3x3(x2)
        x3 = self.inception_double_3x3_reduce(x)
        x3 = self.inception_double_3x3_a(x3)
        x3 = self.inception_double_3x3_b(x3)
        x4 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        out = [x2, x3, x4]
        return torch.cat(out, 1)


class googlenet_bn(nn.Module):
    def __init__(self, num_class=1000):
        super(googlenet_bn, self).__init__()
        # print('trace1')
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # print('trace2')
        self.conv2_reduce = BasicConv2d(64, 64, kernel_size=1, stride=1)
        self.conv2 = BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)

        self.inception_3a = Inception(nc_in=192, nc_1x1=64, nc_3x3_reduce=64, nc_3x3=64, nc_double_3x3_reduce=64,
                                      nc_double_3x3_a=96, nc_double_3x3_b=96, nc_pool_conv=32)
        # print('trace3')
        self.inception_3b = Inception(nc_in=256, nc_1x1=64, nc_3x3_reduce=64, nc_3x3=96, nc_double_3x3_reduce=64,
                                      nc_double_3x3_a=96, nc_double_3x3_b=96, nc_pool_conv=64)
        self.inception_3c = Inception_downsample(nc_in=320, nc_3x3_reduce=128, nc_3x3=160, nc_double_3x3_reduce=64,
                                                 nc_double_3x3_a=96, nc_double_3x3_b=96)
        self.inception_4a = Inception(nc_in=576, nc_1x1=224, nc_3x3_reduce=64, nc_3x3=96, nc_double_3x3_reduce=96,
                                      nc_double_3x3_a=128, nc_double_3x3_b=128, nc_pool_conv=128)
        self.inception_4b = Inception(nc_in=576, nc_1x1=192, nc_3x3_reduce=96, nc_3x3=128, nc_double_3x3_reduce=96,
                                      nc_double_3x3_a=128, nc_double_3x3_b=128, nc_pool_conv=128)
        self.inception_4c = Inception(nc_in=576, nc_1x1=160, nc_3x3_reduce=128, nc_3x3=160, nc_double_3x3_reduce=128,
                                      nc_double_3x3_a=160, nc_double_3x3_b=160, nc_pool_conv=96)
        self.inception_4d = Inception(nc_in=576, nc_1x1=96, nc_3x3_reduce=128, nc_3x3=192, nc_double_3x3_reduce=160,
                                      nc_double_3x3_a=192, nc_double_3x3_b=192, nc_pool_conv=96)
        self.inception_4e = Inception_downsample(nc_in=576, nc_3x3_reduce=128, nc_3x3=192, nc_double_3x3_reduce=192,
                                                 nc_double_3x3_a=256, nc_double_3x3_b=256)
        self.inception_5a = Inception(nc_in=1024, nc_1x1=352, nc_3x3_reduce=192, nc_3x3=320, nc_double_3x3_reduce=160,
                                      nc_double_3x3_a=224, nc_double_3x3_b=224, nc_pool_conv=128)
        self.inception_5b = Inception(nc_in=1024, nc_1x1=352, nc_3x3_reduce=192, nc_3x3=320, nc_double_3x3_reduce=192,
                                      nc_double_3x3_a=224, nc_double_3x3_b=224, nc_pool_conv=128)
        self.classifier = nn.Linear(1024, num_class)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')

    def forward(self, x):
        # print('x',x.size)
        x = self.conv1(x)
        # print('conv1',x.size())
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        # print('maxpool1',x.size())
        x = self.conv2_reduce(x)
        x = self.conv2(x)
        # print('conv2',x.size())
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        # print('maxpool2',x.size())
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.inception_3c(x)
        # print('inception3',x.size())
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        # print('inception4',x.size())
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        # print('inception5',x.size())
        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.squeeze()
        # print('avgpool',x.size())
        x = self.classifier(x)
        return x