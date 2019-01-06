from os import path
import pickle
from glob import glob
import re
import numpy as np


def get_inference_result(train_time):
    """
    :param train_time: the train time you want to evaluate inference result
    :return:
    """
    base_path = '/data/fhz/unsupervised_recommendation'
    feature_path = path.join(base_path, "vae_inference", "train_time_{}".format(train_time))
    if not path.exists(feature_path):
        raise FileNotFoundError("{} not exist".format(feature_path))
    with open(path.join(feature_path, "train.pkl"), "rb") as pkl_file:
        train_feature = pickle.load(pkl_file)
    with open(path.join(feature_path, "test.pkl"), "rb") as pkl_file:
        test_feature = pickle.load(pkl_file)
    parameter_path = glob(path.join(base_path, "vae_parameter",
                                    "unsupervised_recommendation_train_vae_time_{}*".format(train_time)))[0]
    dataset_name = re.match("(.*)_train_dset_(.*)", path.basename(parameter_path)).group(2)
    latent_dim = train_feature[list(train_feature.keys())[0]]["mu"].shape[1]
    train_dname_list = list(train_feature.keys())
    train_mu_vec = np.empty((len(train_dname_list), latent_dim), dtype=np.float)
    train_log_sigma_vec = np.empty((len(train_dname_list), latent_dim), dtype=np.float)
    train_label = list([])
    for idx, dname in enumerate(train_dname_list):
        train_label.append(int(eval(re.match("(.*)_(.*)_annotations", dname).group(2)) > 3))
        train_mu_vec[idx, ...] = train_feature[dname]["mu"]
        train_log_sigma_vec[idx, ...] = train_feature[dname]["log_sigma"]
    train_label = np.array(train_label)

    test_dname_list = list(test_feature.keys())
    test_mu_vec = np.empty((len(test_dname_list), latent_dim), dtype=np.float)
    test_log_sigma_vec = np.empty((len(test_dname_list), latent_dim), dtype=np.float)
    test_label = list([])
    for idx, dname in enumerate(test_dname_list):
        test_label.append(int(eval(re.match("(.*)_(.*)_annotations", dname).group(2)) > 3))
        test_mu_vec[idx, ...] = test_feature[dname]["mu"]
        test_log_sigma_vec[idx, ...] = test_feature[dname]["log_sigma"]
    test_label = np.array(test_label)
    return train_label, train_mu_vec, train_log_sigma_vec, train_dname_list, test_label, test_mu_vec, test_log_sigma_vec, test_dname_list


def get_ae_inference_result(train_time):
    """
    :param train_time: the train time you want to evaluate inference result
    :return:
    """
    base_path = '/data/fhz/unsupervised_recommendation'
    feature_path = path.join(base_path, "ae_inference", "train_time_{}".format(train_time))
    if not path.exists(feature_path):
        raise FileNotFoundError("{} not exist".format(feature_path))
    with open(path.join(feature_path, "train.pkl"), "rb") as pkl_file:
        train_feature = pickle.load(pkl_file)
    with open(path.join(feature_path, "test.pkl"), "rb") as pkl_file:
        test_feature = pickle.load(pkl_file)
    parameter_path = glob(path.join(base_path, "ae_parameter",
                                    "unsupervised_recommendation_train_ae_time_{}*".format(train_time)))[0]
    dataset_name = re.match("(.*)_train_dset_(.*)", path.basename(parameter_path)).group(2)
    latent_dim = train_feature[list(train_feature.keys())[0]]["z"].shape[1]
    train_dname_list = list(train_feature.keys())
    train_z_vec = np.empty((len(train_dname_list), latent_dim), dtype=np.float)
    train_label = list([])
    for idx, dname in enumerate(train_dname_list):
        train_label.append(int(eval(re.match("(.*)_(.*)_annotations", dname).group(2)) > 3))
        train_z_vec[idx, ...] = train_feature[dname]["z"]
    train_label = np.array(train_label)

    test_dname_list = list(test_feature.keys())
    test_z_vec = np.empty((len(test_dname_list), latent_dim), dtype=np.float)
    test_label = list([])
    for idx, dname in enumerate(test_dname_list):
        test_label.append(int(eval(re.match("(.*)_(.*)_annotations", dname).group(2)) > 3))
        test_z_vec[idx, ...] = test_feature[dname]["z"]
    test_label = np.array(test_label)
    return train_label, train_z_vec, train_dname_list, test_label, test_z_vec, test_dname_list
