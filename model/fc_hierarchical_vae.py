import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _LatentVariableInference2d(nn.Module):
    def __init__(self, input_feature_num, latent_dim, output_feature_num):
        super(_LatentVariableInference2d, self).__init__()
        self.z_mean_map = nn.Linear(input_feature_num, latent_dim)
        self.z_log_sigma_map = nn.Linear(input_feature_num, latent_dim)
        self.z_decoder_map = nn.Linear(latent_dim, output_feature_num)

    def _inference(self, features):
        mu = self.z_mean_map(features)
        log_sigma = self.z_log_sigma_map(features)
        sigma = torch.exp(log_sigma)
        self.z_mean = mu
        self.z_sigma = sigma
        self.z_log_sigma = log_sigma
        return mu, log_sigma, sigma

    def _sample_latent(self, features):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu, log_sigma, sigma = self._inference(features)
        std_z = torch.randn(sigma.size())
        if features.is_cuda:
            std_z = std_z.cuda()
        return mu + sigma * std_z

    def decode_latent_variable(self, z):
        decode_features = self.z_decoder_map(z)
        return decode_features

    def forward(self, features):
        z = self._sample_latent(features)
        decode_features = self.decode_latent_variable(z)
        return decode_features, self.z_mean, self.z_sigma, self.z_log_sigma


class _FullyConnectionBlock(nn.Sequential):
    def __init__(self, input_feature_num, output_feature_num, drop_rate):
        super(_FullyConnectionBlock, self).__init__()
        self.add_module("linear", nn.Linear(input_feature_num, output_feature_num))
        self.add_module('norm', nn.BatchNorm1d(output_feature_num))
        self.add_module('relu', nn.ReLU())
        self.add_module("drop", nn.Dropout(p=drop_rate))


class FcHiearachicalVAE(nn.Module):
    def __init__(self, num_input_channels=1, hidden_unit_config=[400, 400], drop_rate=float(0), latent_dim=[8, 8],
                 img_size=[368, 464], data_parallel=True):
        """
        :param num_input_channels: the channel of images to imput
        :param hidden_unit_config: the config for hidden unity
        :param img_size: the size for image
        :param latent_dim: the list for latent dim
        :param drop_rate: the drop rate for network
        """
        super(FcHiearachicalVAE, self).__init__()
        self.img_size = tuple(img_size)
        self.encoder = nn.Sequential()
        self.inference = nn.Sequential()
        self.decoder = nn.Sequential()
        self.encoder.add_module("fc1", _FullyConnectionBlock(
            input_feature_num=int(num_input_channels * img_size[0] * img_size[1]),
            output_feature_num=hidden_unit_config[0], drop_rate=drop_rate))
        self.inference.add_module("inf1", _LatentVariableInference2d(input_feature_num=hidden_unit_config[0],
                                                                     latent_dim=latent_dim[0],
                                                                     output_feature_num=hidden_unit_config[0]))
        self.encoder.add_module("fc2", _FullyConnectionBlock(
            input_feature_num=hidden_unit_config[0],
            output_feature_num=hidden_unit_config[1], drop_rate=drop_rate))
        self.inference.add_module("inf2", _LatentVariableInference2d(input_feature_num=hidden_unit_config[1],
                                                                     latent_dim=latent_dim[0],
                                                                     output_feature_num=hidden_unit_config[1]))

        self.decoder.add_module("fc2", _FullyConnectionBlock(
            input_feature_num=hidden_unit_config[1],
            output_feature_num=hidden_unit_config[0], drop_rate=drop_rate))
        self.decoder.add_module("fc1", _FullyConnectionBlock(
            input_feature_num=hidden_unit_config[0],
            output_feature_num=int(num_input_channels * img_size[0] * img_size[1]), drop_rate=drop_rate))
        for name, param in self.named_parameters():
            if ('linear' in name or 'map' in name) and 'weight' in name:
                nn.init.xavier_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

    def forward(self, input_img):
        img_size = input_img.size()
        input_img = input_img.view(input_img.size(0), -1)
        encode_f1 = self.encoder.fc1(input_img)
        inference_features_1, z_mean_1, z_sigma_1, z_log_sigma_1 = self.inference.inf1(encode_f1)
        encode_f2 = self.encoder.fc2(encode_f1)
        inference_features_2, z_mean_2, z_sigma_2, z_log_sigma_2 = self.inference.inf1(encode_f2)
        decode_f1 = self.decoder.fc2(inference_features_2)
        reconstruct_img = self.decoder.fc1((decode_f1 + inference_features_1))
        reconstruct_img = reconstruct_img.view(img_size)
        z_mean_list = [z_mean_1, z_mean_2]
        z_sigma_list = [z_sigma_1, z_sigma_2]
        z_log_sigma_list = [z_log_sigma_1, z_log_sigma_2]
        return reconstruct_img, z_mean_list, z_sigma_list, z_log_sigma_list

    def generate_from_raw_img(self, input_img, generate_times=1):
        """
        :param input_img: 1*channel*h*w
        :param generate_times:time to generate the reconstruct images
        :return: generate img with size generate_times * channel * h * w
        """
        with torch.no_grad():
            repeat_img = input_img.repeat(generate_times, 1, 1, 1)
            reconstruct_img, latent_variable_mean_list, latent_variable_sigma_list, latent_variable_log_sigma_list = self.forward(
                repeat_img)
        return reconstruct_img, latent_variable_mean_list, latent_variable_sigma_list, latent_variable_log_sigma_list

    def generate_from_latent_variable(self, z_list):
        """
        :param z_list: a list to store the sampled hierarchical z variable,example:
        [latent_layer_1_z,latent_layer_2_z,...,latent_layer_L,z ]
        the size of latent_layer_i_z is batch_size * latent_dim
        :return: batch_size * img_size reconstructed images
        """
        with torch.no_grad():
            inference_feature2 = self.inference.inf2.decode_latent_variable(z_list[1])
            inference_feature1 = self.inference.inf1.decode_latent_variable(z_list[0])
            decode_f1 = self.decoder.fc2(inference_feature2)
            reconstruct_img = self.decoder.fc1((decode_f1 + inference_feature1))
            batch_size = z_list[0].shape[0]
            reconstruct_img = reconstruct_img.view(batch_size, 1, self.img_size[0], self.img_size[1])
        return reconstruct_img


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    network = FcHiearachicalVAE(latent_dim=[8, 8])
    network = network.cuda()
    input_data = torch.randn(4, 1, 368, 464)
    input_data = input_data.cuda()
    reconstruct_img, latent_variable_mean_list, latent_variable_sigma_list, latent_variable_log_sigma_list = network(
        input_data)
    print(reconstruct_img.size(), latent_variable_mean_list, latent_variable_log_sigma_list)
    output = network.generate_from_latent_variable(z_list=latent_variable_mean_list)
    print(output.size())
