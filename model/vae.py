"""
vae is short for variational auto encoder
"""
import __init__
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import DenseNet2d, DenseNet3d
from decoder import DenseDecoder3D, DenseDecoder2D
import numpy as np


class VAE3d(nn.Module):
    r"""VAE3d is based on the DenseNet3d and DenseDecoder3D
    Args:
        feature_dim : the dim for features
        input_channel,small_inputs: parameters in DenseNet3d
    """

    def __init__(self, latent_space_dim=128, input_channel=1, small_inputs=True, initial_feature_size=list([4, 6, 6])):
        super(VAE3d, self).__init__()
        self.encoder = DenseNet3d(input_channel=input_channel, small_inputs=small_inputs)
        self.z_mean_map = nn.Linear(384 * (initial_feature_size[0] * initial_feature_size[1] * initial_feature_size[2]),
                                    latent_space_dim)
        self.z_log_sigma_map = nn.Linear(
            384 * (initial_feature_size[0] * initial_feature_size[1] * initial_feature_size[2]), latent_space_dim)
        self.decoder = DenseDecoder3D(initial_feature_size=initial_feature_size, latent_feature_size=latent_space_dim,
                                      small_inputs=small_inputs)

    def _sample_latent(self, features):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self.z_mean_map(features)
        log_sigma = self.z_log_sigma_map(features)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        std_z = std_z.cuda()
        self.z_mean = mu
        self.z_sigma = sigma
        self.z_log_sigma = log_sigma
        return mu + sigma * std_z

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(x.size(0), -1)
        z = self._sample_latent(features)
        x_reconstructed = self.decoder(z)
        return x_reconstructed


class VAE2d(nn.Module):
    r"""VAE3d is based on the DenseNet3d and DenseDecoder3D
    Args:
        feature_dim : the dim for features
        input_channel,small_inputs: parameters in DenseNet3d
    """

    def __init__(self, latent_space_dim=128, input_channel=3, small_inputs=False, initial_feature_size=[8, 12]):
        super(VAE2d, self).__init__()
        self.encoder = DenseNet2d(input_channel=input_channel, small_inputs=small_inputs)
        self.z_mean_map = nn.Linear(384 * (initial_feature_size[0] * initial_feature_size[1]), latent_space_dim)
        self.z_log_sigma_map = nn.Linear(384 * (initial_feature_size[0] * initial_feature_size[1]), latent_space_dim)
        self.decoder = DenseDecoder2D(initial_feature_size=initial_feature_size, latent_feature_size=latent_space_dim,
                                      small_inputs=small_inputs, num_input_channels=input_channel)

    def _sample_latent(self, features):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self.z_mean_map(features)
        log_sigma = self.z_log_sigma_map(features)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()
        std_z = std_z.cuda()
        self.z_mean = mu
        self.z_sigma = sigma
        self.z_log_sigma = log_sigma
        return mu + sigma * std_z

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
    # vae3d = VAE3d()
    # vae3d = vae3d.cuda()
    # x = torch.randn(2, 1, 32, 48, 48)
    # x = x.cuda()
    # x_reconstructed = vae3d(x)
    # print(x_reconstructed.size())
    vae2d = VAE2d()
    vae2d=vae2d.cuda()
    x = torch.randn(2, 3,512,768)
    x = x.cuda()
    x_reconstructed = vae2d(x)
    print(x_reconstructed.size())
