import __init__
from model import fc_hierarchical_vae
from model import denseunet_hierarchical_vae
from model import ladder_vae
from lib.Criterion import VAECriterion
from torch import nn
from lib.utils.avgmeter import AverageMeter
from lib.dataloader import SGCCDataset
from torch.utils.data import DataLoader
import os
import torch
from glob import glob
from os import path
import time
import shutil
import numpy as np
import re
from random import sample
from tensorboardX import SummaryWriter
from copy import copy
from statistics import mean
from lib.filter import prob_sim_matrix, spectral_k_cut
import pickle
import ast
from lib.utils.crossval import multi_cross_validation
from random import sample
import cv2

fold = 1
dataset_path = path.join("/data/fhz", "sgcc_dataset")
data_path_list = glob(path.join(dataset_path, "*.png"))
train_data_path_list, test_data_path_list = multi_cross_validation(data_path_list, fold)

sample_data = sample(test_data_path_list, 64)
raw_img = []
for data_path in sample_data:
    image = cv2.imread(data_path, 0)
    image_name = path.basename(data_path)[:-4]
    image = np.expand_dims(image, 0)
    image = (image - 255) / 255
    raw_img.append(image)
raw_img = np.concatenate(raw_img, axis=0)
raw_img_cuda = torch.from_numpy(raw_img).unsqueeze(1).float().cuda()
store_folder_path = "/data/fhz/vae_sample"
if not path.exists(store_folder_path):
    os.makedirs(store_folder_path)
np.save(path.join(store_folder_path, "raw_img.npy"), raw_img)


resume_path = "/data/fhz/sgcc_vae/vae_parameter/train_vae_time_13_train_dset_sgcc_dataset/fold_{}/model_test_total_loss_best.pth.tar".format(
    fold)
checkpoint = torch.load(resume_path)
model = denseunet_hierarchical_vae.DenseUnetHiearachicalVAE(latent_dim=checkpoint["args"].latent_dim,
                                                            data_parallel=checkpoint["args"].data_parallel,
                                                            img_size=checkpoint["args"].image_size,
                                                            block_config=checkpoint["args"].block_config).cuda()
model.load_state_dict(checkpoint['state_dict'])
model = model.eval()
reconstruct_img_cuda, *_ = model(raw_img_cuda)
reconstruct_img = reconstruct_img_cuda.detach().cpu().squeeze().numpy()
np.save(path.join(store_folder_path, "denseu_hvae.npy"), reconstruct_img)

resume_path = "/data/fhz/sgcc_lvae/vae_parameter/train_vae_time_1_train_dset_sgcc_dataset/fold_{}/model_test_total_loss_best.pth.tar".format(
    fold)
checkpoint = torch.load(resume_path)
model = ladder_vae.LVAE(latent_dim=checkpoint["args"].latent_dim, img_size=checkpoint["args"].image_size,
                        hidden_unit_config=checkpoint["args"].hidden_config)
model = model.cuda()
model.load_state_dict(checkpoint['state_dict'])
model = model.eval()
reconstruct_img_cuda, *_ = model(raw_img_cuda)
reconstruct_img = reconstruct_img_cuda.detach().cpu().squeeze().numpy()
np.save(path.join(store_folder_path, "ladder_vae.npy"), reconstruct_img)

resume_path = "/data/fhz/sgcc_fcvae/vae_parameter/train_vae_time_1_train_dset_sgcc_dataset/fold_{}/model_test_total_loss_best.pth.tar".format(
    fold)
checkpoint = torch.load(resume_path)
model = fc_hierarchical_vae.FcHiearachicalVAE(latent_dim=checkpoint["args"].latent_dim,
                                              img_size=checkpoint["args"].image_size,
                                              hidden_unit_config=checkpoint["args"].hidden_config)
model = model.cuda()
model.load_state_dict(checkpoint['state_dict'])
model = model.eval()
reconstruct_img_cuda, *_ = model(raw_img_cuda)
reconstruct_img = reconstruct_img_cuda.detach().cpu().squeeze().numpy()
np.save(path.join(store_folder_path, "fc_vae.npy"), reconstruct_img)