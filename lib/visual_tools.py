import __init__
import numpy as np
import cv2
from glob import glob
import os
from os import path
import math
from tensorboardX import SummaryWriter
import pickle
import re
from utils.crossval import multi_cross_validation
import torch
from dataloader import SGCCDataset
from torch.utils.data import DataLoader
from scipy import ndimage
import shutil
import matplotlib.pyplot as plt
from utils.calculate_dist import gaussian_kl_calculation_vec_pairwise
from utils.normalize_hu import cut_and_normalize_hu, normalize_hu

plt.switch_backend('agg')
from pylab import subplots_adjust
from utils.inference_result import get_inference_result, get_ae_inference_result
from random import sample


def visual2dsgcc(train_time, dataset="sgcc_dataset",  target_epoch=1159, save_folder="/data/fhz/sgcc_vae/vae_result_visual"):
    """
    :param train_time: the time of training
    :param save_folder: the folder path to save the image
    :return: it will generate .png type file under the
    """
    save_folder = path.join(save_folder,
                            "train_vae_time_{}_train_dset_{}".format(train_time, dataset, type))
    if path.exists(save_folder):
        flag = input("The folder {} already have files, enter yes to clear it:".format(path.basename(save_folder)))
        if flag == "yes":
            shutil.rmtree(save_folder)
    if not path.exists(save_folder):
        os.makedirs(save_folder)
    base_path = "/data/fhz/sgcc_vae/train_vae_time_{}_train_dset_{}/test_{}_epoch".format(
        train_time, dataset,  target_epoch)
    base_path = path.join(base_path, os.listdir(base_path)[0])
    for target_img_path in glob(path.join(base_path, "*")):
        reconstruct_npy_list = glob(path.join(target_img_path, "recons*.npy"))
        raw_path = path.join(target_img_path, "raw.npy")
        fig = plt.figure(dpi=4000)
        subplots_adjust(hspace=1.2, wspace=0.2)
        img = np.squeeze(np.load(raw_path))
        subfig = fig.add_subplot(3, 4, 1)
        subfig.set_title("raw_img")
        subfig.imshow(img,cmap="gray")
        for idx, img_path in enumerate(reconstruct_npy_list):
            rcl, kl = re.match("reconstruct_(.*)_rcl_(.*)_kl(.*).npy", path.basename(img_path)).group(2, 3)
            img = np.squeeze(np.load(img_path))
            subfig = fig.add_subplot(3, 4, (idx + 2))
            subfig.set_title("rcl:{:.1f}\n kl:{:.1f}".format(eval(rcl), eval(kl)))
            subfig.imshow(img, cmap="gray")
        fig.savefig(path.join(save_folder, '{}.png'.format(path.basename(target_img_path))))
        plt.close()
        print("The result of instance {}'s visualization finished".format(path.basename(target_img_path)))


def visual_mean_feature(train_time):
    """
    Here mean feature means the mean value of latent space the vae predicts
    :param train_time : the parameter of train time
    """
    base_path = '/data/fhz/unsupervised_recommendation'
    visual_path = path.join(base_path, "vae_mean_feature_visual")
    if not path.exists(visual_path):
        os.makedirs(visual_path)

    log_dir = path.join(visual_path, "train_time_{}".format(train_time))
    if path.exists(log_dir):
        flag = input("{} will be removed, input yes to continue:".format(
            log_dir))
        if flag == "yes":
            shutil.rmtree(log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=log_dir)
    # Here we load the data set with their features
    feature_path = path.join(base_path, "vae_inference", "train_time_{}".format(train_time))
    if not path.exists(feature_path):
        raise FileNotFoundError("{} not exist".format(feature_path))
    with open(path.join(feature_path, "train.pkl"), "rb") as pkl_file:
        train_feature = pickle.load(pkl_file)
    with open(path.join(feature_path, "test.pkl"), "rb") as pkl_file:
        test_feature = pickle.load(pkl_file)
    # Here we create a tensor to store the train_image and test_image as well as their label and mean_features
    # actually for the saving point don't save the train and test information, so here we
    # need to calculate it as a replacement
    parameter_path = glob(path.join(base_path, "vae_parameter",
                                    "unsupervised_recommendation_train_vae_time_{}*".format(train_time)))[0]
    dataset_name = re.match("(.*)_train_dset_(.*)", path.basename(parameter_path)).group(2)
    latent_dim = train_feature[list(train_feature.keys())[0]]["mu"].shape[1]
    """
    Here is the elegant code:
    parameter_path = glob(
        path.join(base_path, "vae_parameter", "unsupervised_recommendation_train_vae_time_{}*".format(train_time)))[
        0]
    checkpoint = torch.load(path.join(parameter_path, "checkpoint.pth.tar"))
    dataset_name = checkpoint["dataset"]
    train_datalist = checkpoint["train_datalist"]
    test_datalist = checkpoint["test_datalist"]
    latent_dim = checkpoint["latent_dim"]
    del checkpoint
    """
    if dataset_name == "lung":
        # Here we use it for temptation
        train_datalist, test_datalist = multi_cross_validation()
        train_dataset = LungDataSet(train_datalist, need_name_label=True, need_malignancy_label=True)
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1)
        train_image = torch.randn(len(train_datalist), 1, 32, 32)
        train_mean_feature = torch.randn(len(train_datalist), latent_dim)
        # train_label = [int(eval(re.match("(.*)_(.*)_annotations.npy", path.basename(raw_cube_path)).group(2)) > 3)
        #                for raw_cube_path in train_datalist]
        train_label = list([])
        for i, (img, idx, img_name, malignancy, *_) in enumerate(train_dataloader):
            img = torch.squeeze(img).float()
            img_name, *_ = img_name
            malignancy, *_ = malignancy
            print(img.size(), img_name, int(malignancy))
            train_image[i, ...] = torch.from_numpy(ndimage.zoom(img[16, ...].numpy(), (2 / 3, 2 / 3), mode="nearest"))
            train_mean_feature[i, ...] = torch.from_numpy(train_feature[img_name]["mu"])
            train_label.append(int(malignancy))
        writer.add_embedding(train_mean_feature, metadata=train_label, label_img=train_image, tag="train",
                             global_step=1)
        print("Mean Feature of TrainDset Visual Finished")
        if test_datalist is not None:
            test_dataset = LungDataSet(test_datalist, need_name_label=True, need_malignancy_label=True)
            test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1)
            test_image = torch.randn(len(test_datalist), 1, 32, 32)
            test_mean_feature = torch.randn(len(test_datalist), latent_dim)
            # test_label = [int(eval(re.match("(.*)_(.*)_annotations.npy", path.basename(raw_cube_path)).group(2)) > 3)
            #               for raw_cube_path in test_datalist]
            test_label = list([])
            for i, (img, idx, img_name, malignancy, *_) in enumerate(test_dataloader):
                img = torch.squeeze(img).float()
                img_name, *_ = img_name
                malignancy, *_ = malignancy
                print(img.size(), img_name, int(malignancy))
                test_image[i, ...] = torch.from_numpy(
                    ndimage.zoom(img[16, ...].numpy(), (2 / 3, 2 / 3), mode="nearest"))
                test_mean_feature[i, ...] = torch.from_numpy(test_feature[img_name]["mu"])
                test_label.append(int(malignancy))
            writer.add_embedding(test_mean_feature, metadata=test_label, label_img=test_image, tag="test",
                                 global_step=2)
            print("Mean Feature of TestDset Visual Finished")
            # Here we also add an all embeddings
            all_labels = train_label + test_label
            dataset_label = ["train"] * len(train_label) + ["test"] * len(test_label)
            all_labels = list(zip(all_labels, dataset_label))
            all_features = torch.cat((train_mean_feature, test_mean_feature))
            all_image = torch.cat((train_image, test_image))
            writer.add_embedding(all_features, metadata=all_labels, label_img=all_image,
                                 metadata_header=['label', 'dataset'], global_step=3, tag="all")
            print("Mean Feature of Train&Test Dataset Visual Finished")
        else:
            exit("test data list is none")
    elif dataset_name == "gland":
        dataset_path = "/data/fhz/MICCAI2015/npy"
        train_datalist = glob(path.join(dataset_path, "train", "*.npy"))
        test_datalist = glob(path.join(dataset_path, "test", "*.npy"))

        exit("Here I haven't finish gland part")
    else:
        raise NameError("Dataset {} not found".format(dataset_name))


def visual_similarity(train_time, sample_train_num=20, sample_test_num=10, measure_type="kl"):
    save_folder = path.join("/data/fhz/unsupervised_recommendation/vae_similarity_visual",
                            "train_time_{}_{}_similarity_result".format(train_time, measure_type))
    if path.exists(save_folder):
        flag = input("The folder {} already have files, enter yes to clear it:".format(path.basename(save_folder)))
        if flag == "yes":
            shutil.rmtree(save_folder)
    if not path.exists(save_folder):
        os.makedirs(save_folder)
    train_label, train_mu_vec, train_log_sigma_vec, train_dname_list, test_label, test_mu_vec, test_log_sigma_vec, \
    test_dname_list = get_inference_result(train_time=train_time)
    if measure_type == "kl":
        distance_matrix_train2train = gaussian_kl_calculation_vec_pairwise(u1=train_mu_vec,
                                                                           log_sigma1=train_log_sigma_vec,
                                                                           u2=train_mu_vec,
                                                                           log_sigma2=train_log_sigma_vec,
                                                                           GPU_flag=False)
        distance_matrix_train2test = gaussian_kl_calculation_vec_pairwise(u1=train_mu_vec,
                                                                          log_sigma1=train_log_sigma_vec,
                                                                          u2=test_mu_vec, log_sigma2=test_log_sigma_vec,
                                                                          GPU_flag=False)
        distance_matrix_test2train = gaussian_kl_calculation_vec_pairwise(u2=train_mu_vec,
                                                                          log_sigma2=train_log_sigma_vec,
                                                                          u1=test_mu_vec, log_sigma1=test_log_sigma_vec,
                                                                          GPU_flag=False)
        distance_matrix_test2test = gaussian_kl_calculation_vec_pairwise(u2=test_mu_vec, log_sigma2=test_log_sigma_vec,
                                                                         u1=test_mu_vec, log_sigma1=test_log_sigma_vec,
                                                                         GPU_flag=False)
    elif measure_type == "euclidean":
        n_train, dim = train_mu_vec.shape
        n_test, dim = test_mu_vec.shape
        distance_matrix_train2train = np.sum(
            (train_mu_vec.reshape(n_train, 1, dim) - train_mu_vec.reshape(1, n_train, dim)) ** 2, axis=2)
        distance_matrix_train2test = np.sum(
            (train_mu_vec.reshape(n_train, 1, dim) - test_mu_vec.reshape(1, n_test, dim)) ** 2, axis=2)
        distance_matrix_test2train = np.sum(
            (test_mu_vec.reshape(n_test, 1, dim) - train_mu_vec.reshape(1, n_train, dim)) ** 2, axis=2)
        distance_matrix_test2test = np.sum(
            (test_mu_vec.reshape(n_test, 1, dim) - test_mu_vec.reshape(1, n_test, dim)) ** 2, axis=2)
    else:
        raise NotImplementedError("measure {} not implemented".format(measure_type))
    tag = "train2train"
    idx_list = sample(range(len(train_dname_list)), sample_train_num)
    for sample_id in idx_list:
        kl_tensor = torch.from_numpy(distance_matrix_train2train[sample_id, :])
        y_dist, y_i = kl_tensor.topk(6, dim=0, largest=False, sorted=True)
        data_name_list = [train_dname_list[i] for i in y_i]
        label_list = [train_label[i] for i in y_i]
        visual_similarity_cube(data_name_list=data_name_list, dist_list=y_dist, label_list=label_list,
                               save_folder=save_folder, tag=tag,
                               measure_type=measure_type)
    tag = "train2test"
    idx_list = sample(range(len(train_dname_list)), sample_train_num)
    for sample_id in idx_list:
        kl_tensor = torch.from_numpy(distance_matrix_train2test[sample_id, :])
        y_dist, y_i = kl_tensor.topk(5, dim=0, largest=False, sorted=True)
        data_name_list = [test_dname_list[i] for i in y_i]
        label_list = [test_label[i] for i in y_i]
        data_name_list.insert(0, train_dname_list[sample_id])
        label_list.insert(0, train_label[sample_id])
        y_dist = torch.cat((torch.DoubleTensor([0.0]), y_dist))
        visual_similarity_cube(data_name_list=data_name_list, dist_list=y_dist, label_list=label_list,
                               save_folder=save_folder, tag=tag,
                               measure_type=measure_type)
    tag = "test2train"
    idx_list = sample(range(len(test_dname_list)), sample_test_num)
    for sample_id in idx_list:
        kl_tensor = torch.from_numpy(distance_matrix_test2train[sample_id, :])
        y_dist, y_i = kl_tensor.topk(5, dim=0, largest=False, sorted=True)
        data_name_list = [train_dname_list[i] for i in y_i]
        label_list = [train_label[i] for i in y_i]
        data_name_list.insert(0, test_dname_list[sample_id])
        label_list.insert(0, test_label[sample_id])
        y_dist = torch.cat((torch.DoubleTensor([0.0]), y_dist))
        visual_similarity_cube(data_name_list=data_name_list, dist_list=y_dist, label_list=label_list,
                               save_folder=save_folder, tag=tag,
                               measure_type=measure_type)
    tag = "test2test"
    idx_list = sample(range(len(test_dname_list)), sample_test_num)
    for sample_id in idx_list:
        kl_tensor = torch.from_numpy(distance_matrix_test2test[sample_id, :])
        y_dist, y_i = kl_tensor.topk(6, dim=0, largest=False, sorted=True)
        data_name_list = [test_dname_list[i] for i in y_i]
        label_list = [test_label[i] for i in y_i]
        visual_similarity_cube(data_name_list=data_name_list, dist_list=y_dist, label_list=label_list,
                               save_folder=save_folder, tag=tag,
                               measure_type=measure_type)


def visual_ae_similarity(train_time, sample_train_num=20, sample_test_num=10, measure_type="euclidean"):
    save_folder = path.join("/data/fhz/unsupervised_recommendation/ae_similarity_visual",
                            "train_time_{}_{}_similarity_result".format(train_time, measure_type))
    if path.exists(save_folder):
        flag = input("The folder {} already have files, enter yes to clear it:".format(path.basename(save_folder)))
        if flag == "yes":
            shutil.rmtree(save_folder)
    if not path.exists(save_folder):
        os.makedirs(save_folder)
    train_label, train_z_vec, train_dname_list, test_label, test_z_vec, test_dname_list = get_ae_inference_result(
        train_time=train_time)
    if measure_type == "euclidean":
        n_train, dim = train_z_vec.shape
        n_test, dim = test_z_vec.shape
        distance_matrix_train2train = np.sum(
            (train_z_vec.reshape(n_train, 1, dim) - train_z_vec.reshape(1, n_train, dim)) ** 2, axis=2)
        distance_matrix_train2test = np.sum(
            (train_z_vec.reshape(n_train, 1, dim) - test_z_vec.reshape(1, n_test, dim)) ** 2, axis=2)
        distance_matrix_test2train = np.sum(
            (test_z_vec.reshape(n_test, 1, dim) - train_z_vec.reshape(1, n_train, dim)) ** 2, axis=2)
        distance_matrix_test2test = np.sum(
            (test_z_vec.reshape(n_test, 1, dim) - test_z_vec.reshape(1, n_test, dim)) ** 2, axis=2)
    else:
        raise NotImplementedError("measure {} not implemented".format(measure_type))
    tag = "train2train"
    idx_list = sample(range(len(train_dname_list)), sample_train_num)
    for sample_id in idx_list:
        distance = torch.from_numpy(distance_matrix_train2train[sample_id, :])
        y_dist, y_i = distance.topk(6, dim=0, largest=False, sorted=True)
        data_name_list = [train_dname_list[i] for i in y_i]
        label_list = [train_label[i] for i in y_i]
        visual_similarity_cube(data_name_list=data_name_list, dist_list=y_dist, label_list=label_list,
                               save_folder=save_folder, tag=tag,
                               measure_type=measure_type)
    tag = "train2test"
    idx_list = sample(range(len(train_dname_list)), sample_train_num)
    for sample_id in idx_list:
        distance = torch.from_numpy(distance_matrix_train2test[sample_id, :])
        y_dist, y_i = distance.topk(5, dim=0, largest=False, sorted=True)
        data_name_list = [test_dname_list[i] for i in y_i]
        label_list = [test_label[i] for i in y_i]
        data_name_list.insert(0, train_dname_list[sample_id])
        label_list.insert(0, train_label[sample_id])
        y_dist = torch.cat((torch.DoubleTensor([0.0]), y_dist))
        visual_similarity_cube(data_name_list=data_name_list, dist_list=y_dist, label_list=label_list,
                               save_folder=save_folder, tag=tag,
                               measure_type=measure_type)
    tag = "test2train"
    idx_list = sample(range(len(test_dname_list)), sample_test_num)
    for sample_id in idx_list:
        distance = torch.from_numpy(distance_matrix_test2train[sample_id, :])
        y_dist, y_i = distance.topk(5, dim=0, largest=False, sorted=True)
        data_name_list = [train_dname_list[i] for i in y_i]
        label_list = [train_label[i] for i in y_i]
        data_name_list.insert(0, test_dname_list[sample_id])
        label_list.insert(0, test_label[sample_id])
        y_dist = torch.cat((torch.DoubleTensor([0.0]), y_dist))
        visual_similarity_cube(data_name_list=data_name_list, dist_list=y_dist, label_list=label_list,
                               save_folder=save_folder, tag=tag,
                               measure_type=measure_type)
    tag = "test2test"
    idx_list = sample(range(len(test_dname_list)), sample_test_num)
    for sample_id in idx_list:
        distance = torch.from_numpy(distance_matrix_test2test[sample_id, :])
        y_dist, y_i = distance.topk(6, dim=0, largest=False, sorted=True)
        data_name_list = [test_dname_list[i] for i in y_i]
        label_list = [test_label[i] for i in y_i]
        visual_similarity_cube(data_name_list=data_name_list, dist_list=y_dist, label_list=label_list,
                               save_folder=save_folder, tag=tag,
                               measure_type=measure_type)


def visual_similarity_cube(data_name_list, dist_list, label_list, save_folder, measure_type="kl",
                           base_path="/data/fhz/lidc_cubes_64_overbound/npy", tag="train2train"):
    """
    :param data_name_list: the list to store the data name, always:[target_cube,1st similar cube, 2nd similar cube,
    3rd ...] , the length is always 6.
    :param dist_list: array or list to store the dist,length is the length of data path list ,with 0 first
    :param label_list: array or list to store the label
    :param measure_type: "kl divergence" or "euclidean" or "cosine" or others
    :param save_folder: the folder to save result
    :param base_path: the path to store the original data npy
    :return: nothing but save pictures
    """
    if not path.exists(save_folder):
        raise FileNotFoundError("{} is not a folder".format(save_folder))
    data_path_list = [path.join(base_path, data_name + ".npy") for data_name in data_name_list]
    fig = plt.figure(dpi=4000)
    subplots_adjust(hspace=1.2, wspace=0.2)
    raw_cube = np.squeeze(np.load(data_path_list[0]))
    raw_cube = cut_and_normalize_hu(raw_cube)
    raw_flat_cube = trans_cube_2_flat(cube=raw_cube)
    subfig = fig.add_subplot(3, 2, 1)
    subfig.set_title("raw_img \n label :{}".format(label_list[0]))
    subfig.imshow(raw_flat_cube, cmap="gray")
    for idx in range(1, len(data_path_list)):
        dist = dist_list[idx]
        label = label_list[idx]
        sim_cube = np.squeeze(np.load(data_path_list[idx]))
        sim_cube = cut_and_normalize_hu(sim_cube)
        sim_flat_cube = trans_cube_2_flat(cube=sim_cube)
        subfig = fig.add_subplot(3, 2, (idx + 1))
        subfig.set_title("sim rank :{:.1f}\n dist {}:{:.1f} label:{}".format(idx, measure_type, dist, label))
        subfig.imshow(sim_flat_cube, cmap="gray")
    raw_cube_name = path.basename(data_path_list[0])[:-4]
    fig.savefig(path.join(save_folder, 'tag:{}_{}.png'.format(tag, raw_cube_name)))
    plt.close()
    print("The result of instance {}'s similarity visualization finished".format(raw_cube_name))


if __name__ == "__main__":
    # folder_path = "/data/fhz/unsupervised_recommendation/unsupervised_recommendation_train_vae_time_1_train_dset_lung/test_899_epoch/total_kl_reconstruct_loss_best"
    # target_dir_path_list = [os.path.join(folder_path, dir_name) for dir_name in os.listdir(folder_path)]
    # save_dir_name = "flat_image"
    # for target_dir_path in target_dir_path_list:
    #     visual3dcube(target_dir_path, save_dir_name)
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # for i in range(14, 16):
    #     visual3dcube(train_time=i)
    # visual3dcube(train_time=16)
    # visual_similarity(train_time=16)
    # visual2dsgcc(dataset="25000Img",train_time=12,target_epoch=239)
    visual2dsgcc(dataset="sgcc_dataset",train_time=10,target_epoch=1199)
