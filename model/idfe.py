"""
idfe is short for instance discrimination feature extractor
"""
import __init__
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from encoder import DenseNet2d, DenseNet3d
from decoder import DenseDecoder3D, DenseDecoder2D


class IdFe3d(DenseNet3d):
    r"""IdFe3d is based on the DenseNet3d in encoder module
    Args:
        feature_dim : the dim for features
        input_channel,small_inputs: parameters in DenseNet3d
    """

    def __init__(self, feature_dim=128, input_channel=1, small_inputs=True):
        super(IdFe3d, self).__init__(input_channel=input_channel, small_inputs=small_inputs)
        self.linear_map = nn.Linear(384 * (4 ** 3), feature_dim)

    def forward(self, x):
        out_features = self.encoder(x)
        out = F.relu(out_features, inplace=True)
        out = out.view(out_features.size(0), -1)
        out = F.normalize(self.linear_map(out))
        return out


class IdFe2d(DenseNet2d):
    r"""IdFe3d is based on the DenseNet3d in encoder module
    Args:
        feature_dim : the dim for features
        input_channel,small_inputs: parameters in DenseNet3d
    """

    def __init__(self, feature_dim=128, input_channel=3, small_inputs=False):
        super(IdFe2d, self).__init__(input_channel=input_channel, small_inputs=small_inputs)
        self.linear_map = nn.Linear(384 * (8*12), feature_dim)

    def forward(self, x):
        out_features = self.encoder(x)
        out = F.relu(out_features, inplace=True)
        out = out.view(out_features.size(0), -1)
        out = F.normalize(self.linear_map(out))
        return out


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    network = IdFe2d()
    network = network.cuda()
    input_data = torch.randn(4, 3, 522, 775)
    input_data = input_data.cuda()
    features = network(input_data)
    print(features.size())
