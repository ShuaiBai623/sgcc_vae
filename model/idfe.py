"""
idfe is short for instance discrimination feature extractor
"""
import __init__
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from encoder import DenseNet2d


class IdFe2d(DenseNet2d):
    r"""IdFe3d is based on the DenseNet3d in encoder module
    Args:
        feature_dim : the dim for features
        input_channel,small_inputs: parameters in DenseNet3d
    """

    def __init__(self, num_input_channels=1, growth_rate=12, block_config=(6, 12, 24, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=float(0), efficient=False, latent_dim=128,
                 img_size=(368, 464), data_parallel=True):
        super(IdFe2d, self).__init__(num_input_channels, growth_rate, block_config, compression, num_init_features,
                                     bn_size, drop_rate, efficient)
        img_size = [int(s / 2 ** (2 + len(block_config) - 1)) for s in img_size]
        channel_number_list = self.calculate_channel_number(num_init_features, block_config, growth_rate, compression)
        self.linear_map = nn.Linear(img_size[0] * img_size[1] * channel_number_list[-1], latent_dim)
        if data_parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.linear_map = nn.DataParallel(self.linear_map)
        for name, param in self.linear_map.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    def forward(self, x):
        out = self.encoder(x)
        out = out.view(out.size(0), -1)
        out = F.normalize(self.linear_map(out))
        return out

    @staticmethod
    def calculate_channel_number(num_init_features, block_config, growth_rate, compression):
        """
        :param num_init_features: the init channel number for dense encoder
        :param block_config: the dense block layer config for dense encoder
        :param growth_rate: the growth rate for dense layer
        :param compression: the compression rate (default :0.5)
        :param final_compression_flag: if true, then after final dense block, we will do bn,relu conv and maxpooling,
        which means we will do channel compress (in conv operation). if false, then we only do bn
        :return: channel_number_list
        """
        channel = num_init_features
        channel_number_list = []
        for i, layer_num in enumerate(block_config):
            channel += growth_rate * layer_num
            channel *= compression
            channel_number_list.append(int(channel))
        # in final transition we don't do compression
        channel_number_list[-1] = int(channel_number_list[-1] / compression)
        return channel_number_list


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    network = IdFe2d(latent_dim=16)
    network = network.cuda()
    input_data = torch.randn(1, 1, 368, 464)
    input_data = input_data.cuda()
    features = network(input_data)
    print(features.size())
