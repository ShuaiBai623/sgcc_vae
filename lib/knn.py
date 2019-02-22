import torch
import numpy as np
import os
import __init__
from utils.calculate_dist import pairwise_norm_wasserstein_dist_gpu, pairwise_square_euclidean_gpu
from utils.crossval import multi_cross_validation
from os import path
import pickle
from random import sample
from collections import defaultdict


# def unsupervised_map(fold, train_time, base_path, model, k, gamma, GPU_flag=True):
#     model_list = ["instance_discrimination", "denseu_hvae", "fc_hvae", "ladder_vae"]
#     label_dict_path = path.join(base_path, "sgcc_dataset", "label.pkl")
#     with open(label_dict_path, "rb") as fp:
#         label_dict = pickle.load(fp)
#     if model == "instance_discrimination":
#         inference_dict_path = path.join(base_path, "sgcc_idfe", "idfe_inference",
#                                         "train_time_{}_dset_sgcc_dataset".format(train_time), "fold_{}".format(fold))
#         with open(path.join(inference_dict_path, "train.pkl"), "rb") as fp:
#             train_dict = pickle.load(fp)
#         with open(path.join(inference_dict_path, "test.pkl"), "rb") as fp:
#             test_dict = pickle.load(fp)
#         with open(path.join(inference_dict_path, "pattern.pkl"), "rb") as fp:
#             pattern_dict = pickle.load(fp)
#         train_dataname_list = list(train_dict.keys())
#         test_dataname_list = list(test_dict.keys())
#         pattern_dataname_list = list(pattern_dict.keys())
#         feature_length = train_dict[train_dataname_list[0]]["feature"].shape[1]
#         annotated_train_feature = np.random.randn(len(train_dataname_list), feature_length)
#         annotated_train_label = np.random.randn(len(train_dataname_list), 11)
#         annotated_test_feature = np.random.randn(len(test_dataname_list), feature_length)
#         annotated_test_label = np.random.randn(len(test_dataname_list), 11)
#         pattern_feature = np.random.randn(len(pattern_dataname_list),11)
#         for idx, name in train_dataname_list:
#             annotated_train_feature[idx, :] = train_dict[name]["feature"]
#             annotated_train_label[idx, :] = np.array(label_dict[name])
#         for idx, name in test_dataname_list:
#             annotated_test_feature[idx, :] = test_dict[name]["feature"]
#             annotated_test_label[idx, :] = np.array(label_dict[name])
#         for idx,name in
#         # notice -1 * similarity is dist
#         dist = -1 * annotated_train_feature.dot(annotated_test_feature.T)
#         if GPU_flag:
#             dist = dist.cuda()
#         per_class_result_dict = defaultdict(dict)
#         for i in range(11):
#             probs, predictions, TP, TN, FP, FN = kNN_one_label(dist, train_label=annotated_train_label[:, i],
#                                                                test_label=annotated_test_label[:, i], k=k, gamma=gamma)
#             per_class_result_dict[str(i)]["prob_matrix"] = probs
#             per_class_result_dict[str(i)]["predictions"] = predictions
#             per_class_result_dict[str(i)]["confusion_matrix"] = [TP, TN, FP, FN]
def semi_supervised_knn(fold, train_time, base_path, annotated_rate, model, k, gamma, GPU_flag=True):
    """
    :param fold: which fold to do knn
    :param train_time: which train_time we use
    :param base_path: the base path of dataset
    :param annotated_rate: how many data in train dataset will be annotated
    :param model: which model we use, we have
    instance_discrimination,denseu_hvae,fc_hvae,ladder_vae 4 parts
    :param k : the k for knn
    :return:gamma: the gamma for knn
    """
    model_list = ["instance_discrimination", "denseu_hvae", "fc_hvae", "ladder_vae"]
    if model not in model_list:
        raise NotImplementedError("model {} not implemented".format(model))
    label_dict_path = path.join(base_path, "sgcc_dataset", "label.pkl")
    with open(label_dict_path, "rb") as fp:
        label_dict = pickle.load(fp)
    if model == "instance_discrimination":
        inference_dict_path = path.join(base_path, "sgcc_idfe", "idfe_inference",
                                        "train_time_{}_dset_sgcc_dataset".format(train_time), "fold_{}".format(fold))
        label_dict_path = path.join(base_path, "sgcc_dataset", "label.pkl")
        with open(path.join(inference_dict_path, "train.pkl"), "rb") as fp:
            train_dict = pickle.load(fp)
        with open(path.join(inference_dict_path, "test.pkl"), "rb") as fp:
            test_dict = pickle.load(fp)
        with open(path.join(inference_dict_path, "pattern.pkl"), "rb") as fp:
            pattern_dict = pickle.load(fp)
        train_dataname_list = list(train_dict.keys())
        test_dataname_list = list(test_dict.keys())
        sample_num = int(len(train_dataname_list) * annotated_rate)
        annotated_train_data = sample(train_dataname_list, sample_num)
        feature_length = train_dict[annotated_train_data[0]]["feature"].shape[1]
        annotated_train_feature = np.random.randn(sample_num, feature_length)
        annotated_train_label = np.random.randn(sample_num, 11)
        annotated_test_feature = np.random.randn(len(test_dataname_list), feature_length)
        annotated_test_label = np.random.randn(len(test_dataname_list), 11)
        for idx, name in enumerate(annotated_train_data):
            annotated_train_feature[idx, :] = train_dict[name]["feature"]
            annotated_train_label[idx, :] = np.array(label_dict[name])
        for idx, name in enumerate(test_dataname_list):
            annotated_test_feature[idx, :] = test_dict[name]["feature"]
            annotated_test_label[idx, :] = np.array(label_dict[name])
        # notice -1 * similarity is dist
        dist = -1 * annotated_test_feature.dot(annotated_train_feature.T)
        if GPU_flag:
            dist = torch.from_numpy(dist).cuda()
        per_class_result_dict = defaultdict(dict)
        for i in range(11):
            probs, predictions, TP, TN, FP, FN = kNN_one_label(dist, train_label=annotated_train_label[:, i],
                                                               test_label=annotated_test_label[:, i], k=k, gamma=gamma)
            per_class_result_dict[str(i)]["gt"] = annotated_test_label[:, i]
            per_class_result_dict[str(i)]["prob_matrix"] = probs
            per_class_result_dict[str(i)]["predictions"] = predictions
            per_class_result_dict[str(i)]["confusion_matrix"] = [TP, TN, FP, FN]
    elif model == "denseu_hvae":
        inference_dict_path = path.join(base_path, "sgcc_vae", "vae_inference",
                                        "train_time_{}_dset_sgcc_dataset".format(train_time), "fold_{}".format(fold))
        with open(path.join(inference_dict_path, "train.pkl"), "rb") as fp:
            train_dict = pickle.load(fp)
        with open(path.join(inference_dict_path, "test.pkl"), "rb") as fp:
            test_dict = pickle.load(fp)
        with open(path.join(inference_dict_path, "pattern.pkl"), "rb") as fp:
            pattern_dict = pickle.load(fp)
        train_dataname_list = list(train_dict.keys())
        test_dataname_list = list(test_dict.keys())
        sample_num = int(len(train_dataname_list) * annotated_rate)
        annotated_train_data = sample(train_dataname_list, sample_num)
        latent_variable_length = np.concatenate(train_dict[annotated_train_data[0]]["mu"], axis=1).shape[1]
        annotated_train_mu = np.random.randn(sample_num, latent_variable_length)
        annotated_train_log_sigma = np.random.randn(sample_num, latent_variable_length)
        annotated_train_label = np.random.randn(sample_num, 11)
        annotated_test_mu = np.random.randn(len(test_dataname_list), latent_variable_length)
        annotated_test_log_sigma = np.random.randn(len(test_dataname_list), latent_variable_length)
        annotated_test_label = np.random.randn(len(test_dataname_list), 11)
        for idx, name in enumerate(annotated_train_data):
            annotated_train_mu[idx, :] = np.concatenate(train_dict[name]["mu"], axis=1)
            annotated_train_log_sigma[idx, :] = np.concatenate(train_dict[name]["log_sigma"], axis=1)
            annotated_train_label[idx, :] = np.array(label_dict[name])
        for idx, name in enumerate(test_dataname_list):
            annotated_test_mu[idx, :] = np.concatenate(test_dict[name]["mu"], axis=1)
            annotated_test_log_sigma[idx, :] = np.concatenate(test_dict[name]["log_sigma"], axis=1)
            annotated_test_label[idx, :] = np.array(label_dict[name])
        # notice -1 * similarity is dist
        if GPU_flag:
            annotated_test_mu = torch.from_numpy(annotated_test_mu).cuda()
            annotated_test_log_sigma = torch.from_numpy(annotated_test_log_sigma).cuda()
            annotated_train_mu = torch.from_numpy(annotated_train_mu).cuda()
            annotated_train_log_sigma = torch.from_numpy(annotated_train_log_sigma).cuda()
        dist = pairwise_norm_wasserstein_dist_gpu(u1=annotated_test_mu, log_sigma1=annotated_test_log_sigma,
                                                  u2=annotated_train_mu, log_sigma2=annotated_train_log_sigma)
        #         dist = torch.from_numpy(dist).cuda()
        per_class_result_dict = defaultdict(dict)
        for i in range(11):
            probs, predictions, TP, TN, FP, FN = kNN_one_label(dist, train_label=annotated_train_label[:, i],
                                                               test_label=annotated_test_label[:, i], k=k, gamma=gamma)
            per_class_result_dict[str(i)]["gt"] = annotated_test_label[:, i]
            per_class_result_dict[str(i)]["prob_matrix"] = probs
            per_class_result_dict[str(i)]["predictions"] = predictions
            per_class_result_dict[str(i)]["confusion_matrix"] = [TP, TN, FP, FN]
    elif model == "fc_hvae":
        inference_dict_path = path.join(base_path, "sgcc_fcvae", "vae_inference",
                                        "train_time_{}_dset_sgcc_dataset".format(train_time), "fold_{}".format(fold))
        with open(path.join(inference_dict_path, "train.pkl"), "rb") as fp:
            train_dict = pickle.load(fp)
        with open(path.join(inference_dict_path, "test.pkl"), "rb") as fp:
            test_dict = pickle.load(fp)
        with open(path.join(inference_dict_path, "pattern.pkl"), "rb") as fp:
            pattern_dict = pickle.load(fp)
        train_dataname_list = list(train_dict.keys())
        test_dataname_list = list(test_dict.keys())
        sample_num = int(len(train_dataname_list) * annotated_rate)
        annotated_train_data = sample(train_dataname_list, sample_num)
        latent_variable_length = np.concatenate(train_dict[annotated_train_data[0]]["mu"], axis=1).shape[1]
        annotated_train_mu = np.random.randn(sample_num, latent_variable_length)
        annotated_train_log_sigma = np.random.randn(sample_num, latent_variable_length)
        annotated_train_label = np.random.randn(sample_num, 11)
        annotated_test_mu = np.random.randn(len(test_dataname_list), latent_variable_length)
        annotated_test_log_sigma = np.random.randn(len(test_dataname_list), latent_variable_length)
        annotated_test_label = np.random.randn(len(test_dataname_list), 11)
        for idx, name in enumerate(annotated_train_data):
            annotated_train_mu[idx, :] = np.concatenate(train_dict[name]["mu"], axis=1)
            annotated_train_log_sigma[idx, :] = np.concatenate(train_dict[name]["log_sigma"], axis=1)
            annotated_train_label[idx, :] = np.array(label_dict[name])
        for idx, name in enumerate(test_dataname_list):
            annotated_test_mu[idx, :] = np.concatenate(test_dict[name]["mu"], axis=1)
            annotated_test_log_sigma[idx, :] = np.concatenate(test_dict[name]["log_sigma"], axis=1)
            annotated_test_label[idx, :] = np.array(label_dict[name])
        # notice -1 * similarity is dist
        if GPU_flag:
            annotated_test_mu = torch.from_numpy(annotated_test_mu).cuda()
            annotated_test_log_sigma = torch.from_numpy(annotated_test_log_sigma).cuda()
            annotated_train_mu = torch.from_numpy(annotated_train_mu).cuda()
            annotated_train_log_sigma = torch.from_numpy(annotated_train_log_sigma).cuda()
        dist = pairwise_norm_wasserstein_dist_gpu(u1=annotated_test_mu, log_sigma1=annotated_test_log_sigma,
                                                  u2=annotated_train_mu, log_sigma2=annotated_train_log_sigma)
        #         dist = torch.from_numpy(dist).cuda()
        per_class_result_dict = defaultdict(dict)
        for i in range(11):
            probs, predictions, TP, TN, FP, FN = kNN_one_label(dist, train_label=annotated_train_label[:, i],
                                                               test_label=annotated_test_label[:, i], k=k, gamma=gamma)
            per_class_result_dict[str(i)]["gt"] = annotated_test_label[:, i]
            per_class_result_dict[str(i)]["prob_matrix"] = probs
            per_class_result_dict[str(i)]["predictions"] = predictions
            per_class_result_dict[str(i)]["confusion_matrix"] = [TP, TN, FP, FN]
    elif model == "ladder_vae":
        inference_dict_path = path.join(base_path, "sgcc_lvae", "vae_inference",
                                        "train_time_{}_dset_sgcc_dataset".format(train_time), "fold_{}".format(fold))
        with open(path.join(inference_dict_path, "train.pkl"), "rb") as fp:
            train_dict = pickle.load(fp)
        with open(path.join(inference_dict_path, "test.pkl"), "rb") as fp:
            test_dict = pickle.load(fp)
        with open(path.join(inference_dict_path, "pattern.pkl"), "rb") as fp:
            pattern_dict = pickle.load(fp)
        train_dataname_list = list(train_dict.keys())
        test_dataname_list = list(test_dict.keys())
        sample_num = int(len(train_dataname_list) * annotated_rate)
        annotated_train_data = sample(train_dataname_list, sample_num)
        latent_variable_length = np.concatenate(train_dict[annotated_train_data[0]]["mu"], axis=1).shape[1]
        annotated_train_mu = np.random.randn(sample_num, latent_variable_length)
        annotated_train_log_sigma = np.random.randn(sample_num, latent_variable_length)
        annotated_train_label = np.random.randn(sample_num, 11)
        annotated_test_mu = np.random.randn(len(test_dataname_list), latent_variable_length)
        annotated_test_log_sigma = np.random.randn(len(test_dataname_list), latent_variable_length)
        annotated_test_label = np.random.randn(len(test_dataname_list), 11)
        for idx, name in enumerate(annotated_train_data):
            annotated_train_mu[idx, :] = np.concatenate(train_dict[name]["mu"], axis=1)
            annotated_train_log_sigma[idx, :] = np.concatenate(train_dict[name]["log_sigma"], axis=1)
            annotated_train_label[idx, :] = np.array(label_dict[name])
        for idx, name in enumerate(test_dataname_list):
            annotated_test_mu[idx, :] = np.concatenate(test_dict[name]["mu"], axis=1)
            annotated_test_log_sigma[idx, :] = np.concatenate(test_dict[name]["log_sigma"], axis=1)
            annotated_test_label[idx, :] = np.array(label_dict[name])
        # notice -1 * similarity is dist
        if GPU_flag:
            annotated_test_mu = torch.from_numpy(annotated_test_mu).cuda()
            annotated_test_log_sigma = torch.from_numpy(annotated_test_log_sigma).cuda()
            annotated_train_mu = torch.from_numpy(annotated_train_mu).cuda()
            annotated_train_log_sigma = torch.from_numpy(annotated_train_log_sigma).cuda()
        dist = pairwise_norm_wasserstein_dist_gpu(u1=annotated_test_mu, log_sigma1=annotated_test_log_sigma,
                                                  u2=annotated_train_mu, log_sigma2=annotated_train_log_sigma)
        #         dist = torch.from_numpy(dist).cuda()
        per_class_result_dict = defaultdict(dict)
        for i in range(11):
            probs, predictions, TP, TN, FP, FN = kNN_one_label(dist, train_label=annotated_train_label[:, i],
                                                               test_label=annotated_test_label[:, i], k=k, gamma=gamma)
            per_class_result_dict[str(i)]["gt"] = annotated_test_label[:, i]
            per_class_result_dict[str(i)]["prob_matrix"] = probs
            per_class_result_dict[str(i)]["predictions"] = predictions
            per_class_result_dict[str(i)]["confusion_matrix"] = [TP, TN, FP, FN]
    else:
        raise NotImplementedError("model {} not implemented".format(model))
    return per_class_result_dict


def kNN_one_label(dist, train_label, test_label, k, gamma, GPU_Flag=True):
    """
    :param dist: the matrix of distance with test 2 train
    :param train_label: the label of train dataset,always N*1
    :param test_label: the label of test dataset, always N*1
    :param k: the kNN nearest neighbor k
    :param gamma: the gamma of the dist
    :param GPU_Flag: True then we use GPU to accelerate the progress
    :return: classification score, predict result, accuracy
    """
    if type(dist) == np.ndarray:
        dist = torch.from_numpy(dist).float()
    if GPU_Flag:
        dist = dist.cuda()
    yd, yi = dist.topk(k, dim=1, largest=False, sorted=True)
    if type(train_label) == list:
        train_label = np.array(train_label)
        test_label = np.array(test_label)
    train_label = torch.from_numpy(train_label).long()
    test_label = torch.from_numpy(test_label).long()
    test_num = dist.size(0)
    C = train_label.max() + 1
    if C == 1:
        C = 2
    candidates = train_label.view(1, -1).expand(test_num, -1).cuda()
    retrieval = torch.gather(candidates, 1, yi)
    retrieval_one_hot = torch.zeros(k, C).cuda()
    retrieval_one_hot.resize_(test_num * k, C).zero_()
    retrieval_one_hot.scatter_(dim=1, index=retrieval.view(-1, 1), src=torch.tensor(1))
    yd_transform = yd.clone().div_(-1 * gamma).exp_().float()
    probs = torch.sum(
        torch.mul(retrieval_one_hot.view(test_num, -1, C), yd_transform.view(test_num, -1, 1)),
        dim=1)
    _, predictions = probs.sort(1, True)
    predictions = predictions[:, 0].cpu().view(-1, 1)
    test_label = test_label.view(-1, 1)
    TP = torch.sum(predictions * test_label)
    TN = torch.sum((1 - predictions) * (1 - test_label))
    FP = torch.sum(predictions * (1 - test_label))
    FN = torch.sum((1 - predictions) * test_label)
    return probs.detach().cpu().numpy(), predictions.numpy(), TP, TN, FP, FN
