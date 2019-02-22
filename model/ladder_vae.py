import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, in_dim=784, out_dim=512, latent_dim=64):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU()
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(out_dim, latent_dim)
        )
        self.var_layer = nn.Sequential(
            nn.Linear(out_dim, latent_dim),
            # nn.Softplus()
        )

    def forward(self, x):
        h = self.net(x)
        mu, var = self.mean_layer(h), self.var_layer(h)
        var = F.softplus(var) + 1e-8
        return h, mu, var


class FinalDecoder(nn.Module):
    def __init__(self, in_dim=64, out_dim=784, hidden_dim=512):
        super(FinalDecoder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
            # nn.Sigmoid()
        )

    def forward(self, x):
        h = self.net(x)

        return h


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension
    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance
    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    # print("log var1", qv)
    return kl


def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian
    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance
    Return:
        z: tensor: (batch, ...): Samples
    """

    sample = torch.randn(m.shape)
    if m.is_cuda:
        sample = sample.cuda()

    z = m + (v ** 0.5) * sample
    return z


class LVAE(nn.Module):

    def __init__(self, num_input_channels=1, hidden_unit_config=[512, 256], drop_rate=float(0), latent_dim=[8, 8],
                 img_size=[368, 464]):
        super(LVAE,self).__init__()
        self.in_dim = int(num_input_channels * img_size[0] * img_size[1])
        self.layer1_dim = hidden_unit_config[0]
        self.layer2_dim = hidden_unit_config[1]
        self.latent1_dim = latent_dim[0]
        self.latent2_dim = latent_dim[1]

        # initialize required MLPs
        self.MLP1 = MLP(self.in_dim, self.layer1_dim, self.latent1_dim)
        self.MLP2 = MLP(self.layer1_dim, self.layer2_dim, self.latent2_dim)
        self.MLP3 = MLP(self.latent2_dim, self.layer1_dim, self.latent1_dim)
        self.FinalDecoder = FinalDecoder(self.latent1_dim, self.in_dim, self.layer1_dim)

    def forward(self, x):
        x_size = x.size()
        x = x.view(x.size(0), -1)
        encoder_feature_0, q_mu_0_, q_var_0_ = self.MLP1(x)
        encoder_feature_1, q_mu_1, q_var_1 = self.MLP2(encoder_feature_0)
        z_1 = sample_gaussian(q_mu_1, q_var_1)
        decode_feature_0, p_mu_0, p_var_0 = self.MLP3(z_1)
        q_mu_0 = (q_mu_0_ / q_var_0_ + p_mu_0 / p_var_0) / (1 / q_var_0_ + 1 / p_var_0)
        q_var_0 = 1 / (1 / q_var_0_ + 1 / p_var_0)
        z_0 = sample_gaussian(p_mu_0, p_var_0)
        reconstruct_x = self.FinalDecoder(z_0)
        reconstruct_x = reconstruct_x.view(x_size)
        return reconstruct_x, q_mu_0, q_var_0, q_mu_1, q_var_1, p_mu_0, p_var_0

    def generate_from_raw_img(self, input_img, generate_times=1):
        """
        :param input_img: 1*channel*h*w
        :param generate_times:time to generate the reconstruct images
        :return: generate img with size generate_times * channel * h * w
        """
        with torch.no_grad():
            repeat_img = input_img.repeat(generate_times, 1, 1, 1)
            reconstruct_img, q_mu_0, q_var_0, q_mu_1, q_var_1, p_mu_0, p_var_0 = self.forward(repeat_img)
        return reconstruct_img, q_mu_0, q_var_0, q_mu_1, q_var_1, p_mu_0, p_var_0


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    network = LVAE(latent_dim=[8, 8])
    network = network.cuda()
    input_data = torch.randn(4, 1, 368, 464)
    input_data = input_data.cuda()
    reconstruct_img, q_mu_0, q_var_0, q_mu_1, q_var_1, p_mu_0, p_var_0 = network(input_data)
    print(reconstruct_img.size(), q_mu_0, q_var_0, q_mu_1, q_var_1, p_mu_0, p_var_0)
