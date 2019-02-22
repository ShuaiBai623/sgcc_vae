import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from encoder import _DenseBlock2D, _Transition2D
import numpy as np


class _DecoderBlock2D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate=float(0)):
        super(_DecoderBlock2D, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU())
        self.add_module("conv", nn.Conv2d(num_input_features, out_channels=num_output_features,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module("drop", nn.Dropout2d(p=drop_rate))


class _PreProcess(nn.Sequential):
    def __init__(self, num_input_channels, num_init_features):
        super(_PreProcess, self).__init__()
        self.add_module('conv0',
                        nn.Conv2d(num_input_channels, num_init_features, kernel_size=7, stride=2, padding=3,
                                  bias=False))
        self.add_module('norm0', nn.BatchNorm2d(num_init_features))
        self.add_module('relu0', nn.ReLU(inplace=True))
        self.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                              ceil_mode=False))


class _FinalProcess(nn.Sequential):
    def __init__(self, num_input_channels, num_init_features, num_feature_channels, img_size):
        super(_FinalProcess, self).__init__()
        self.add_module("final_norm0", nn.BatchNorm2d(num_feature_channels))
        self.add_module("final_relu0", nn.ReLU())
        self.add_module('final_conv0', nn.Conv2d(num_feature_channels, out_channels=num_init_features,
                                                 kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module("final_deconv0", nn.ConvTranspose2d(num_init_features, out_channels=num_init_features,
                                                            kernel_size=2, stride=2, padding=0, bias=False))
        self.add_module("final_norm1", nn.BatchNorm2d(num_init_features))
        self.add_module("final_relu1", nn.ReLU())
        self.add_module('final_conv1', nn.Conv2d(num_init_features, out_channels=num_input_channels,
                                                 kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module("final_deconv1", nn.ConvTranspose2d(num_input_channels, out_channels=num_input_channels,
                                                            kernel_size=2, stride=2, padding=0, bias=True))
        # self.add_module("interpolate", nn.functional.interpolate(size=img_size, mode='bilinear', align_corners=True))
        # self.add_module("sigmoid", nn.Sigmoid())


class _LatentVariableInference2d(nn.Module):
    def __init__(self, input_size, latent_dim, num_input_channels, num_output_channels):
        super(_LatentVariableInference2d, self).__init__()
        self.input_size = input_size
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.z_mean_map = nn.Linear(int(num_input_channels * input_size[0] * input_size[1]), latent_dim)
        self.z_log_sigma_map = nn.Linear(int(num_input_channels * input_size[0] * input_size[1]), latent_dim)
        self.z_decoder_map = nn.Linear(latent_dim, int(num_output_channels * input_size[0] * input_size[1]))

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
        decode_features = decode_features.view(z.size(0), self.num_output_channels, int(self.input_size[0]),
                                               int(self.input_size[1]))
        return decode_features

    def forward(self, features):
        batch_size = features.size(0)
        features = features.view(batch_size, -1)
        z = self._sample_latent(features)
        decode_features = self.decode_latent_variable(z)
        return decode_features, self.z_mean, self.z_sigma, self.z_log_sigma


class DenseUnetVAE(nn.Module):
    def __init__(self, num_input_channels=1, growth_rate=12, block_config=[6, 12, 24, 16], compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=float(0), efficient=False,
                 latent_dim=16, img_size=[368, 464], data_parallel=True):
        """
        :param num_input_channels: the channel of images to imput
        :param growth_rate: the middle feature for dense block
        :param block_config: the config for block
        :param compression: the compression rate after each dense block
        :param num_init_features:the initialize feature sending to the dense net
        :param bn_size: dense layer's structure is 1*1 and 3*3, bn_size * growth_rate is the middle feature channel
        :param drop_rate: the drop rate for network
        :param efficient: True we use checkpoints for dense layer, false no checkpoints
        """
        super(DenseUnetVAE, self).__init__()
        self.block_config = block_config
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential()
        self.inference = nn.Sequential()
        self.img_size = tuple(img_size)
        pre_process = _PreProcess(num_input_channels, num_init_features)
        if data_parallel:
            pre_process = nn.DataParallel(pre_process)
        self.encoder.add_module("pre_process", pre_process)
        # img_size = [s / 2 for s in img_size]
        img_size = [s / 4 for s in img_size]
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        # add dense block
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
            if data_parallel:
                block = nn.DataParallel(block)
            self.encoder.add_module('denseblock%d' % (i + 1), block)
            num_features = int(num_features + num_layers * growth_rate)
            # Here we notice that for the last dense block, we don't apply transition layer
            if i != len(block_config) - 1:
                trans = _Transition2D(num_input_features=num_features,
                                      num_output_features=int(num_features * compression))
                if data_parallel:
                    trans = nn.DataParallel(trans)
                self.encoder.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
                img_size = [int(s / 2) for s in img_size]

            else:
                trans = nn.BatchNorm2d(num_features)
                if data_parallel:
                    trans = nn.DataParallel(trans)
                self.encoder.add_module('transition%d' % (i + 1), trans)
                inference = _LatentVariableInference2d(input_size=img_size,
                                                       latent_dim=latent_dim,
                                                       num_input_channels=num_features,
                                                       num_output_channels=num_features)
                if data_parallel:
                    inference = nn.DataParallel(inference)
                self.inference.add_module('latent_layer', inference)

        self.decoder_channel_list = self.calculate_channel_number(num_init_features, block_config, growth_rate,
                                                                  compression)
        self.decoder = nn.Sequential(OrderedDict([]))
        for i in range(len(self.decoder_channel_list) - 1, 0, -1):
            decoder_block = _DecoderBlock2D(self.decoder_channel_list[i], self.decoder_channel_list[i - 1],
                                            drop_rate)
            decoder_deconv = nn.ConvTranspose2d(int(self.decoder_channel_list[i - 1]),
                                                out_channels=self.decoder_channel_list[i - 1],
                                                kernel_size=2, stride=2, padding=0, bias=False)
            if data_parallel:
                decoder_block = nn.DataParallel(decoder_block)
                decoder_deconv = nn.DataParallel(decoder_deconv)
            self.decoder.add_module("decoder_block%d" % (i + 1), decoder_block)
            self.decoder.add_module("decoder_block%d_deconv" % (i + 1), decoder_deconv)
        final_process = _FinalProcess(num_input_channels=num_input_channels,
                                      num_init_features=num_init_features,
                                      num_feature_channels=self.decoder_channel_list[0], img_size=self.img_size)
        if data_parallel:
            final_process = nn.DataParallel(final_process)
        self.decoder.add_module("final_process", final_process)
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.xavier_normal_(param.data)
            # initialize liner transform
            elif 'map' in name and 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'map' in name and 'bias' in name:
                param.data.fill_(0)
            # initialize the batch norm layer
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, input_img):
        output = self.encoder.pre_process(input_img)
        inference_feature = None
        latent_variable_mean = None
        latent_variable_sigma = None
        latent_variable_log_sigma = None
        for i in range(len(self.block_config)):
            output = getattr(self.encoder, "denseblock%d" % (i + 1))(output)
            output = getattr(self.encoder, "transition%d" % (i + 1))(output)
            if i == len(self.block_config) - 1:
                inference_feature, latent_variable_mean, latent_variable_sigma, latent_variable_log_sigma = getattr(
                    self.inference, "latent_layer")(
                    output)
        for i in range(len(self.decoder_channel_list), 1, -1):
            if i == len(self.decoder_channel_list):
                output = getattr(self.decoder, "decoder_block%d" % i)(
                    inference_feature)
                output = getattr(self.decoder, "decoder_block%d_deconv" % i)(output)
            else:
                output = getattr(self.decoder, "decoder_block%d" % i)(output)
                output = getattr(self.decoder, "decoder_block%d_deconv" % i)(output)
        output = self.decoder.final_process(output)
        output = torch.sigmoid(F.interpolate(output, size=self.img_size, mode='bilinear', align_corners=True))
        return output, latent_variable_mean, latent_variable_sigma, latent_variable_log_sigma

    def generate_from_raw_img(self, input_img, generate_times=1):
        """
        :param input_img: 1*channel*h*w
        :param generate_times:time to generate the reconstruct images
        :return: generate img with size generate_times * channel * h * w
        """
        with torch.no_grad():
            repeat_img = input_img.repeat(generate_times, 1, 1, 1)
            recon_img, latent_variable_mean, latent_variable_sigma, latent_variable_log_sigma = self.forward(
                repeat_img)
        return recon_img, latent_variable_mean, latent_variable_sigma, latent_variable_log_sigma

    def generate_from_latent_variable(self, z):
        """
        :param z: a numpy array to store the sampled hierarchical z variable,example:
        the size of latent_layer_i_z is batch_size * latent_dim
        :return: batch_size * img_size reconstructed images
        """
        with torch.no_grad():
            if z.size(1) != self.latent_dim:
                raise ValueError("The number of latent dim is not equal to z")
            inference_feature = getattr(self.inference, "latent_layer").module.decode_latent_variable(z)
            for i in range(len(self.decoder_channel_list), 1, -1):
                if i == len(self.decoder_channel_list):
                    output = getattr(self.decoder, "decoder_block%d" % i)(inference_feature)
                    output = getattr(self.decoder, "decoder_block%d_deconv" % i)(output)
                else:
                    output = getattr(self.decoder, "decoder_block%d" % i)(output)
                    output = getattr(self.decoder, "decoder_block%d_deconv" % i)(output)
            output = self.decoder.final_process(output)
            output = torch.sigmoid(F.interpolate(output, size=self.img_size, mode='bilinear', align_corners=True))
        return output

    @staticmethod
    def calculate_channel_number(init_number, block_config, growth_rate, compression, final_compression_flag=False):
        """
        :param init_number: the init channel number for dense encoder
        :param block_config: the dense block layer config for dense encoder
        :param growth_rate: the growth rate for dense layer
        :param compression: the compression rate (default :0.5)
        :param final_compression_flag: if true, then after final dense block, we will do bn,relu conv and maxpooling,
        which means we will do channel compress (in conv operation). if false, then we only do bn
        :return: channel_number_list
        """
        channel = init_number
        channel_number_list = []
        for i, layer_num in enumerate(block_config):
            channel += growth_rate * layer_num
            channel *= compression
            channel_number_list.append(int(channel))
        if not final_compression_flag:
            channel_number_list[-1] = int(channel_number_list[-1] / compression)
        return channel_number_list


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    network = DenseUnetVAE(latent_dim=16)
    network = network.cuda()
    input_data = torch.randn(1, 1, 368, 464)
    input_data = input_data.cuda()
    reconstruct_img, latent_variable_mean, latent_variable_sigma, latent_variable_log_sigma = network(
        input_data)
    print(reconstruct_img.size(), latent_variable_mean, latent_variable_log_sigma)
    output = network.generate_from_latent_variable(z=latent_variable_mean)
    print(output.size())
