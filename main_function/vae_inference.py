import __init__
from model.vae import VAE2d
from model.denseunet_hierarchical_vae import DenseUnetHiearachicalVAE
import os
import torch
import numpy as np


class VAEInference(VAE2d):
    def __init__(self, resume_path, device_id="0,1", gpu_flag=True):
        if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path)
        else:
            raise FileNotFoundError("{} not found".format(resume_path))
        super(VAEInference, self).__init__(latent_space_dim=checkpoint["args"].latent_dim)
        # do data parallel
        self.encoder = torch.nn.DataParallel(self.encoder)
        self.z_log_sigma_map = torch.nn.DataParallel(self.z_log_sigma_map)
        self.z_mean_map = torch.nn.DataParallel(self.z_mean_map)
        self.decoder = torch.nn.DataParallel(self.decoder)
        # do data load
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.z_mean_map.load_state_dict(checkpoint['z_mean_map_state_dict'])
        self.z_log_sigma_map.load_state_dict(checkpoint['z_sigma_map_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if gpu_flag:
            os.environ['CUDA_VISIBLE_DEVICES'] = device_id
            self.cuda()
        self.gpu_flag = gpu_flag

    def get_image_from_original_to_target(self, original, target, split_number, sample_number):
        """
        :param original: the original distribution's parameters e.g.{mu:npy,log_sigma:npy}
        :param target: the target distribution's parameters e.g.{mu:npy,log_sigma:npy}
        :param split_number: the number we should split t=[0,1] in
        :param sample_number: how many items should we sample for each distribution P_t
        :return: numpy array with split_number*sample_number*h*w
        """
        mu_original = original["mu"]
        sigma_original = np.exp(original["log_sigma"])
        mu_target = target["mu"]
        sigma_target = np.exp(target["log_sigma"])
        latent_dim = mu_original.shape[1]
        t_split = np.linspace(0, 1, split_number + 2)[1:split_number + 1].reshape(split_number, 1)
        mu_t = (1 - t_split) * mu_original + t_split * mu_target
        sigma_t = (1 - t_split) * sigma_original + t_split * sigma_target
        norm_sample = np.random.normal(0, 1, size=(sample_number, latent_dim))
        # use the broad cast trick
        z_sample = mu_t.reshape(split_number, 1, latent_dim) + norm_sample.reshape(1, sample_number,
                                                                                   latent_dim) * sigma_t.reshape(
            split_number, 1, latent_dim)
        z_sample = z_sample.reshape(-1, latent_dim)
        reconstruct = self._get_image_by_sample_func(z_sample).squeeze()
        reconstruct = reconstruct.view(split_number, sample_number, reconstruct.size(1),
                                       reconstruct.size(2)).cpu().detach().numpy()
        return reconstruct

    def get_image_change_by_one_bar(self, original_z, latent_variable_index, bar_min, bar_max, split_number):
        """
        :param original_z: the original z sample,latent-dim array
        :param latent_variable_index: the index of latent_variable we want to change
        :param bar_min: the min value of bar
        :param bar_max: the maximum value of bar
        :param split_number: the number we want to split the bar in
        :return: numpy array with split_number * h * w
        """
        z_input = np.repeat(original_z, split_number, axis=0)
        split_array = np.linspace(bar_min, bar_max, split_number)
        z_input[:, latent_variable_index] = split_array
        reconstruct = self._get_image_by_sample_func(z_input)
        return reconstruct.squeeze().cpu().detach().numpy()

    def get_image_by_sample(self, z_sample):
        reconstruct = self._get_image_by_sample_func(z_sample)
        return reconstruct.squeeze().cpu().detach().numpy()

    def _get_image_by_sample_func(self, z_sample):
        """
        :param z_sample: N*latent_dim numpy
        :return: reconstruct image N*h*w
        """
        z_sample = torch.from_numpy(z_sample).float()
        if self.gpu_flag:
            z_sample = z_sample.cuda()
        reconstruct = self.decoder(z_sample)
        return reconstruct


class HVAEInference(DenseUnetHiearachicalVAE):
    def __init__(self, resume_path, device_id="0,1", gpu_flag=True):
        if os.path.isfile(resume_path):
            checkpoint = torch.load(resume_path)
        else:
            raise FileNotFoundError("{} not found".format(resume_path))
        args = checkpoint['args']
        self.latent_dim = args.latent_dim
        if args.dataset == "sgcc_dataset" and args.train_time <= 10:
            super(HVAEInference, self).__init__(latent_dim=args.latent_dim,
                                                data_parallel=args.data_parallel)
        else:
            super(HVAEInference, self).__init__(latent_dim=args.latent_dim,
                                                data_parallel=args.data_parallel,
                                                img_size=args.image_size,
                                                block_config=args.block_config)
        # do data parallel
        self.load_state_dict(checkpoint["state_dict"])
        if gpu_flag:
            os.environ['CUDA_VISIBLE_DEVICES'] = device_id
            self.cuda()
        self.gpu_flag = gpu_flag

    def get_image_from_original_to_target(self, original, target, split_number, sample_number):
        """
        :param original: the original distribution's parameters e.g.{mu:npy,log_sigma:npy}
        :param target: the target distribution's parameters e.g.{mu:npy,log_sigma:npy}
        :param split_number: the number we should split t=[0,1] in
        :param sample_number: how many items should we sample for each distribution P_t
        :return: numpy array with split_number*sample_number*h*w
        """
        mu_original = original["mu"]
        sigma_original = np.exp(original["log_sigma"])
        mu_target = target["mu"]
        sigma_target = np.exp(target["log_sigma"])
        latent_dim = mu_original.shape[1]
        t_split = np.linspace(0, 1, split_number + 2)[1:split_number + 1].reshape(split_number, 1)
        mu_t = (1 - t_split) * mu_original + t_split * mu_target
        sigma_t = (1 - t_split) * sigma_original + t_split * sigma_target

        norm_sample = np.random.normal(0, 1, size=(sample_number, latent_dim))
        # use the broad cast trick
        z_sample = mu_t.reshape(split_number, 1, latent_dim) + norm_sample.reshape(1, sample_number,
                                                                                   latent_dim) * sigma_t.reshape(
            split_number, 1, latent_dim)
        z_sample = z_sample.reshape(-1, latent_dim)
        reconstruct = self.get_image_by_sample(z_sample)
        reconstruct = reconstruct.reshape(split_number, sample_number, reconstruct.shape[1],
                                          reconstruct.shape[2])
        return reconstruct

    def get_image_change_by_one_bar(self, original_z, latent_variable_index, bar_min, bar_max, split_number):
        """
        :param original_z: the original z sample,latent-dim array
        :param latent_variable_index: the index of latent_variable we want to change
        :param bar_min: the min value of bar
        :param bar_max: the maximum value of bar
        :param split_number: the number we want to split the bar in
        :return: numpy array with split_number * h * w
        """
        z_input = np.repeat(original_z, split_number, axis=0)
        split_array = np.linspace(bar_min, bar_max, split_number)
        z_input[:, latent_variable_index] = split_array
        # here z_input is an n*split_number size's latent variable
        reconstruct = self.get_image_by_sample(z_input)
        return reconstruct

    def get_inference_by_image(self, input_image):
        """
        :param input_image should be an batch_size*channel*h*w numpy array
        :return: return mean with [layer1_mean,layer2_mean,layer3_mean], log_sigma with
        [layer1_log_sigma,layer2_log_sigma,layer3_log_sigma],all items in list is an numpy array
        with batch_size * latent_dim
        """
        input_image = torch.from_numpy(input_image).float()
        if self.gpu_flag:
            input_image = input_image.cuda()
        output, latent_variable_mean_list, latent_variable_sigma_list, latent_variable_log_sigma_list = self.forward(
            input_image)
        mean_list = []
        log_sigma_list = []
        for mean_tensor in latent_variable_mean_list:
            mean_list.append(mean_tensor.cpu().detach().numpy())
        for log_sigma_tensor in latent_variable_log_sigma_list:
            log_sigma_list.append(log_sigma_tensor.cpu().detach().numpy())
        return mean_list, log_sigma_list

    def get_image_by_sample(self, z_sample):
        z_sample = torch.from_numpy(z_sample).float()
        if self.gpu_flag:
            z_sample = z_sample.cuda()
        z_sample_list = self._split_sample_into_hierarchical_list(z_sample)
        reconstruct = self.generate_from_latent_variable(z_sample_list)
        return reconstruct.squeeze().cpu().detach().numpy()

    def _split_sample_into_hierarchical_list(self, z_sample):
        z_sample_list = []
        initial_position = 0
        for i in range(len(self.latent_dim)):
            z_sample_list.append(z_sample[:, initial_position:initial_position + self.latent_dim[i]])
            initial_position += self.latent_dim[i]
        return z_sample_list
