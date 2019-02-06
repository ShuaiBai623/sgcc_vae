import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer3D(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
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


class _DeTrainsition3D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate=float(0)):
        super(_DeTrainsition3D, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU())
        self.add_module("conv", nn.Conv3d(num_input_features, out_channels=num_output_features,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module("drop", nn.Dropout3d(p=drop_rate))
        self.add_module("deconv", nn.ConvTranspose3d(num_output_features, out_channels=num_output_features,
                                                     kernel_size=2, stride=2, padding=0))


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
        output_features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*output_features)
            output_features.append(new_features)
        return torch.cat(output_features, 1)


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


class _DeTrainsition2D(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate=float(0)):
        super(_DeTrainsition2D, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU())
        self.add_module("conv", nn.Conv2d(num_input_features, out_channels=num_output_features,
                                          kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module("drop", nn.Dropout2d(p=drop_rate))
        self.add_module("deconv", nn.ConvTranspose2d(num_output_features, out_channels=num_output_features,
                                                     kernel_size=2, stride=2, padding=0))


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
        output_features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*output_features)
            output_features.append(new_features)
        return torch.cat(output_features, 1)


class Decoder2D(nn.Module):
    def __init__(self, initial_feature_size, num_init_features=24, growth_rate=6, encoder_block_config=(6, 12, 24, 16),
                 compression=0.5, drop_rate=float(0), num_input_channels=3, latent_feature_size=128,
                 small_inputs=False):
        """
        :param initial_feature_size: it will be the height and width for feature (waiting for decode),e.g.[8,8]
        :param num_init_features, growth_rate, encoder_block_config, compression: they are all parameters for calculate
        the decoder channels
        :param drop_rate: drop rate, if we use bn
        :param num_input_channels: the image's channels, 2d usually 3
        :param latent_feature_size: the dim for latent space
        :param small_inputs: the flag for if the input is small(will change the final progress)
        """
        super(Decoder2D, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.initial_feature_size = initial_feature_size
        self.decoder_channel_list = self.calculate_channel_number(num_init_features, encoder_block_config, growth_rate,
                                                                  compression)
        self.decoder = nn.Sequential(OrderedDict([]))
        # build the transform from latent space to feature space
        self.latent_linear_map = nn.Linear(latent_feature_size, self.decoder_channel_list[-1] * (
                initial_feature_size[0] * initial_feature_size[1]))
        # self.decoder.add_module("pre_deconv", nn.ConvTranspose2d(self.decoder_channel_list[-1],
        #                                                          out_channels=self.decoder_channel_list[-1],
        #                                                          kernel_size=2, stride=2, padding=0))
        for i in range(len(self.decoder_channel_list) - 1, 0, -1):
            detrans_block = _DeTrainsition2D(self.decoder_channel_list[i], self.decoder_channel_list[i - 1], drop_rate)
            self.decoder.add_module("detrans_block%d" % (len(self.decoder_channel_list) - i), detrans_block)
        # transform the result into [-1,1]
        # 1st conv it to init_feature
        # 2st conv it to num_input_channels
        if small_inputs:
            decode_final_process = nn.Sequential(OrderedDict([
                ("final_norm0", nn.BatchNorm2d(self.decoder_channel_list[0])),
                ("final_relu0", nn.ReLU()),
                ('final_conv0', nn.Conv2d(self.decoder_channel_list[0], out_channels=num_init_features,
                                          kernel_size=3, stride=1, padding=1, bias=False)),
                ("final_deconv0", nn.ConvTranspose2d(num_init_features, out_channels=num_init_features,
                                                     kernel_size=2, stride=2, padding=0)),
                ("final_norm1", nn.BatchNorm2d(num_init_features)),
                ("final_relu1", nn.ReLU()),
                ('final_conv1', nn.Conv2d(num_init_features, out_channels=num_input_channels,
                                          kernel_size=3, stride=1, padding=1, bias=False)),

                ("sigmoid", nn.Sigmoid())
            ]))
        else:
            decode_final_process = nn.Sequential(OrderedDict([
                ("final_norm0", nn.BatchNorm2d(self.decoder_channel_list[0])),
                ("final_relu0", nn.ReLU()),
                ('final_conv0', nn.Conv2d(self.decoder_channel_list[0], out_channels=num_init_features,
                                          kernel_size=3, stride=1, padding=1, bias=False)),
                ("final_deconv0", nn.ConvTranspose2d(num_init_features, out_channels=num_init_features,
                                                     kernel_size=2, stride=2, padding=0)),
                ("final_norm1", nn.BatchNorm2d(num_init_features)),
                ("final_relu1", nn.ReLU()),
                ('final_conv1', nn.Conv2d(num_init_features, out_channels=num_input_channels,
                                          kernel_size=3, stride=1, padding=1, bias=False)),
                ("final_deconv1", nn.ConvTranspose2d(num_input_channels, out_channels=num_input_channels,
                                                     kernel_size=2, stride=2, padding=0))
            ]))
        self.decoder.add_module("final_process", decode_final_process)
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

    def forward(self, input_feature):
        input_feature = self.latent_linear_map(input_feature)
        x = self.decoder(input_feature.view(-1, self.decoder_channel_list[-1], self.initial_feature_size[0],
                                            self.initial_feature_size[1]))
        return x

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


class Decoder3D(nn.Module):
    def __init__(self, initial_feature_size, num_init_features=24, growth_rate=12, encoder_block_config=(6, 12, 24, 16),
                 compression=0.5, drop_rate=float(0), num_input_channels=1, latent_feature_size=128, small_inputs=True):
        """
        :param initial_feature_size: it will be the depth, height and width for feature (waiting for decode),e.g.[8,8,8]
        :param num_init_features, growth_rate, encoder_block_config, compression: they are all parameters for calculate
        the decoder channels
        :param drop_rate: drop rate, if we use bn
        :param num_input_channels: the image's channels, 2d usually 3
        :param latent_feature_size: the dim for latent space
        """
        super(Decoder3D, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        self.initial_feature_size = initial_feature_size
        self.decoder_channel_list = self.calculate_channel_number(num_init_features, encoder_block_config, growth_rate,
                                                                  compression)
        self.decoder = nn.Sequential(OrderedDict([]))
        # build the transform from latent space to feature space
        self.latent_linear_map = nn.Linear(latent_feature_size, self.decoder_channel_list[-1] * (
                initial_feature_size[0] * initial_feature_size[1] * initial_feature_size[2]))
        #
        for i in range(len(self.decoder_channel_list) - 1, 0, -1):
            detrans_block = _DeTrainsition3D(self.decoder_channel_list[i], self.decoder_channel_list[i - 1], drop_rate)
            self.decoder.add_module("detrans_block%d" % (len(self.decoder_channel_list) - i), detrans_block)
        # transform the result into [-1,1]
        # 1st conv it to init_feature
        # 2st conv it to num_input_channels
        if small_inputs:
            decode_final_process = nn.Sequential(OrderedDict([
                ("final_norm0", nn.BatchNorm3d(self.decoder_channel_list[0])),
                ("final_relu0", nn.ReLU()),
                ('final_conv0', nn.Conv3d(self.decoder_channel_list[0], out_channels=num_init_features,
                                          kernel_size=3, stride=1, padding=1, bias=False)),
                # ("final_deconv0", nn.ConvTranspose3d(num_init_features, out_channels=num_init_features,
                #                                      kernel_size=2, stride=2, padding=0)),
                ("final_norm1", nn.BatchNorm3d(num_init_features)),
                ("final_relu1", nn.ReLU()),
                ('final_conv1', nn.Conv3d(num_init_features, out_channels=num_input_channels,
                                          kernel_size=3, stride=1, padding=1, bias=False)),

                ("sigmoid", nn.Sigmoid())
            ]))
        else:
            decode_final_process = nn.Sequential(OrderedDict([
                ("final_norm0", nn.BatchNorm3d(self.decoder_channel_list[0])),
                ("final_relu0", nn.ReLU()),
                ('final_conv0', nn.Conv3d(self.decoder_channel_list[0], out_channels=num_init_features,
                                          kernel_size=3, stride=1, padding=1, bias=False)),
                ("final_deconv0", nn.ConvTranspose3d(num_init_features, out_channels=num_init_features,
                                                     kernel_size=2, stride=2, padding=0)),
                ("final_norm1", nn.BatchNorm3d(num_init_features)),
                ("final_relu1", nn.ReLU()),
                ('final_conv1', nn.Conv3d(num_init_features, out_channels=num_input_channels,
                                          kernel_size=3, stride=1, padding=1, bias=False)),
                ("final_deconv1", nn.ConvTranspose3d(num_input_channels, out_channels=num_input_channels,
                                                     kernel_size=2, stride=2, padding=0)),
                ("sigmoid", nn.Sigmoid())
            ]))
        self.decoder.add_module("final_process", decode_final_process)
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

    def forward(self, input_feature):
        input_feature = self.latent_linear_map(input_feature)

        x = self.decoder(input_feature.view(-1, self.decoder_channel_list[-1], self.initial_feature_size[0],
                                            self.initial_feature_size[1], self.initial_feature_size[2]))
        return x

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
        if final_compression_flag:
            channel_number_list[-1] = int(channel_number_list[-1] / compression)
        return channel_number_list


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # d = DenseNet3d()
    # d = d.cuda()
    # a = torch.randn(4, 1, 64, 64, 64)
    # a = a.cuda()
    # feature = d(a)
    # print(feature.size())
    # network = DenseDecoder2D(initial_feature_size=[8, 8],small_input=False)
    network = Decoder3D(initial_feature_size=[4, 6, 6], small_inputs=True)
    network = network.cuda()
    input_data = torch.randn(4, 128)
    input_data = input_data.cuda()
    features = network(input_data)
    print(features.size())
