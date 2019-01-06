# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

"""
Here we use densenet3D,densenet2D as our encoder network structures
"""


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer3D(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        # note: bn_size is short for bottle neck size, which means we first use 1*1 to reduce
        # the input channel into bn_size * growth_rate
        # and then will use 3*3 kernel to transform the input channel into growth_rate
        super(_DenseLayer3D, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition3D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition3D, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class _DenseBlock3D(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock3D, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer3D(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _DenseLayer2D(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer2D, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition2D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition2D, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock2D(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock2D, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer2D(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet3d(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, input_channel=3, growth_rate=12, block_config=(6, 12, 24, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=float(0), small_inputs=True, efficient=True):

        super(DenseNet3d, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        # First convolution
        if small_inputs:
            self.encoder = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv3d(input_channel, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.encoder = nn.Sequential(OrderedDict([
                ('pre_pool', nn.AvgPool2d(2, 2)),
            ]))
            self.encoder.add_module('conv0',
                                    nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            self.encoder.add_module('norm0', nn.BatchNorm3d(num_init_features))
            self.encoder.add_module('relu0', nn.ReLU(inplace=True))
            self.encoder.add_module('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1,
                                                          ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock3D(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.encoder.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # Here we notice that for the last dense block, we don't apply transition layer
            if i != len(block_config) - 1:
                trans = _Transition3D(num_input_features=num_features,
                                      num_output_features=int(num_features * compression))
                self.encoder.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.encoder.add_module('norm_final', nn.BatchNorm3d(num_features))

        # Initialization( using xavier)
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                # N_in = param.size(0)*kernel_size
                # N_out= param.size(1)*kernel_size
                # variance = 1/N = 1/(N_in + N_out)/2 = 2/(N_in +N_out)
                nn.init.xavier_normal_(param.data)
                # n = param.size(0) * param.size(2) * param.size(3)
                # param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        encoding_features = self.encoder(x)
        return encoding_features


class DenseNet2d(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, input_channel=3, growth_rate=12, block_config=(6, 12, 24, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=float(0), small_inputs=False, efficient=True):

        super(DenseNet2d, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        # First convolution
        if small_inputs:
            self.encoder = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(input_channel, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            """
            it will firstly use convolution with large kernel and stride to down sample image by factor 2,
            then it will use maxpool function with kernel 3 and stride 2 to down sample image by factor 2.
            which means the image will be down sampled by factor 4
            """
            self.encoder = nn.Sequential(OrderedDict([
                ('pre_pool', nn.AvgPool2d(2, 2)),
            ]))
            self.encoder.add_module('conv0',
                                    nn.Conv2d(input_channel, num_init_features, kernel_size=7, stride=2, padding=3,
                                              bias=False))
            self.encoder.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.encoder.add_module('relu0', nn.ReLU(inplace=True))
            self.encoder.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                          ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock2D(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.encoder.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition2D(num_input_features=num_features,
                                      num_output_features=int(num_features * compression))
                self.encoder.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.encoder.add_module('norm_final', nn.BatchNorm2d(num_features))
        # Initialization (by xavier)
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                # N_in = param.size(0)*kernel_size
                # N_out= param.size(1)*kernel_size
                # variance = 1/N = 1/(N_in + N_out)/2 = 2/(N_in +N_out)
                nn.init.xavier_normal_(param.data)
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        encoding_features = self.encoder(x)
        return encoding_features


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # d = DenseNet3d()
    # d = d.cuda()
    # a = torch.randn(4, 1, 64, 64, 64)
    # a = a.cuda()
    # feature = d(a)
    # print(feature.size())
    network = DenseNet3d(input_channel=1,small_inputs=True)
    network = network.cuda()
    input_data = torch.randn(1, 1, 32, 48,48)
    input_data = input_data.cuda()
    features = network(input_data)
    print(features.size())
