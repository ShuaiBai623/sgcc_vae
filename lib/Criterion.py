import torch
from torch import nn
import torch.nn.functional as F

eps = 1e-7


class VAECriterion(nn.Module):
    """
    Here we calculate the VAE loss
    VAE loss's math formulation is :
    E_{z~Q}[log(P(X|z))]-D[Q(z|X)||P(z)]
    which can be transformed into:
    ||X-X_{reconstructed}||^2/(\sigma)^2 - [<L2norm(u)>^2+<L2norm(diag(\Sigma))>^2
    -<L2norm(diag(ln(\Sigma)))>^2-1]
    Our input is :
    x_sigma,x_reconstructed,x,z_mean,z_Sigma
    """

    def __init__(self, x_sigma=1):
        super(VAECriterion, self).__init__()
        self.x_sigma = x_sigma

    def forward(self, x, x_reconstructed, z_mean, z_log_sigma, z_sigma):
        """
        :param x: input & ground truth
        :param x_reconstructed: the reconstructed output by VAE
        :param z_mean: the mean of latent space Q(z|X)
        :param z_sigma: the variance of latent space
        :param z_log_sigma : log(z_sigma)
        :return: reconstruct_loss, kl_loss
        """
        batch_size = x.size(0)
        # calculate reconstruct loss, sum in instance, mean in batch
        reconstruct_loss = F.mse_loss(x_reconstructed, x, reduction="sum")
        reconstruct_loss = reconstruct_loss / (self.x_sigma ** 2 * batch_size)
        # reconstruct_loss = F.mse_loss(x_reconstructed, x)
        # reconstruct_loss = reconstruct_loss / self.x_sigma ** 2
        # calculate latent space KL divergence
        z_mean_sq = z_mean * z_mean
        z_sigma_sq = z_sigma * z_sigma
        z_log_sigma_sq = 2 * z_log_sigma
        kl_loss = torch.sum(z_mean_sq + z_sigma_sq - z_log_sigma_sq - 1) / batch_size
        # kl_loss = 0.5 * torch.mean(z_mean_sq + z_sigma_sq - z_log_sigma_sq - 1)
        return reconstruct_loss, kl_loss


class AECriterion(nn.Module):
    """
    Here we calculate the VAE loss
    VAE loss's math formulation is :
    E_{z~Q}[log(P(X|z))]-D[Q(z|X)||P(z)]
    which can be transformed into:
    ||X-X_{reconstructed}||^2/(\sigma)^2 - [<L2norm(u)>^2+<L2norm(diag(\Sigma))>^2
    -<L2norm(diag(ln(\Sigma)))>^2-1]
    Our input is :
    x_sigma,x_reconstructed,x,z_mean,z_Sigma
    """

    def __init__(self, x_sigma=1):
        super(AECriterion, self).__init__()
        self.x_sigma = x_sigma

    def forward(self, x, x_reconstructed):
        """
        :param x: input & ground truth
        :param x_reconstructed: the reconstructed output by VAE
        :param z_mean: the mean of latent space Q(z|X)
        :param z_sigma: the variance of latent space
        :param z_log_sigma : log(z_sigma)
        :return: reconstruct_loss, kl_loss
        """
        batch_size = x.size(0)
        # calculate reconstruct loss, sum in instance, mean in batch
        reconstruct_loss = F.mse_loss(x_reconstructed, x, reduction="sum")
        reconstruct_loss = reconstruct_loss / (self.x_sigma * batch_size)
        # reconstruct_loss = F.mse_loss(x_reconstructed, x)
        # reconstruct_loss = reconstruct_loss / self.x_sigma ** 2
        # calculate latent space KL divergence
        return reconstruct_loss
