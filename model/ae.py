import __init__
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import DenseNet2d, DenseNet3d
from decoder import DenseDecoder3D, DenseDecoder2D
import numpy as np


class AE3d(nn.Module):
    r"""VAE3d is based on the DenseNet3d and DenseDecoder3D
    Args:
        feature_dim : the dim for features
        input_channel,small_inputs: parameters in DenseNet3d
    """

    def __init__(self, latent_space_dim=128, input_channel=1, small_inputs=True, initial_feature_size=list([4, 6, 6])):
        super(AE3d, self).__init__()
        self.encoder = DenseNet3d(input_channel=input_channel, small_inputs=small_inputs)
        self.z_map = nn.Linear(384 * (initial_feature_size[0] * initial_feature_size[1] * initial_feature_size[2]),
                               latent_space_dim)
        self.decoder = DenseDecoder3D(initial_feature_size=initial_feature_size, latent_feature_size=latent_space_dim,
                                      small_inputs=small_inputs)

    def _sample_latent(self, features):
        z = self.z_map(features)
        self.z = z
        return z

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(x.size(0), -1)
        z = self._sample_latent(features)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # network = VAE2d()
    # network = network.cuda()
    # x = torch.randn(2, 3, 522, 775)
    # x = x.cuda()
    # x_reconstructed, z_mean, z_sigma = network(x)
    # print(x_reconstructed.size(), z_mean.size(), z_sigma.size())
    ae3d = AE3d()
    ae3d = ae3d.cuda()
    x = torch.randn(2, 1, 32, 48, 48)
    x = x.cuda()
    x_reconstructed = ae3d(x)
    print(x_reconstructed.size())