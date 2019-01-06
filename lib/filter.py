import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from collections import defaultdict


def prob_sim_matrix(feature, tau=1):
    """
    :param feature: numpy array
    :param tau: exp(<vi,vj>/tau)
    :return: similarity matrix
    """
    m = np.exp(np.dot(feature, feature.T) / tau)
    return m


def spectral_k_cut(data_list, target_num, feature, represent_select="1", affinity_matrix=None):
    """
    :param data_list: the data list for filter with every element a path
    :param target_num: the num we want to choose
    :param feature: numpy array, recording the feature of each item in data_list
    :param represent_select: the strategy to select, we use max-cover,min-max,centroid point 3 strategy
    :param affinity_matrix: N*N affinity matrix
    :return:
    """
    filtered_datalist = list([])
    spectral_cls_result = SpectralClustering(n_clusters=target_num, affinity="precomputed", n_init=50, n_jobs=4).fit(
        affinity_matrix)
    cls_labels = spectral_cls_result.labels_
    cls_dict = defaultdict(list)
    for index, cls_label in enumerate(cls_labels):
        cls_dict[str(cls_label)].append(index)
    for cls_name, label_list in cls_dict.items():
        if len(label_list) == 1:
            filtered_datalist.append(data_list[label_list[0]])
        else:
            sub_affinity_matrix = affinity_matrix[label_list, :][:, label_list]
            # Here we have 2 possible methods:
            # 1. calculate f_i =sum_j s_ij and target_i = argmax f_i
            # 2. calculate g_i = min s_ij , and target_i = argmax g_i
            # 3. calculate the centroid point of data and find the nearest data
            if represent_select == "1":
                f = np.sum(sub_affinity_matrix, axis=1)
                select_index = np.argmax(f)
            elif represent_select == "2":
                g = np.min(sub_affinity_matrix, axis=1)
                select_index = np.argmax(g)
            elif represent_select == "3":
                sub_feature = feature[label_list, :]
                centroid = np.mean(sub_feature, axis=0)
                distance = np.sum((sub_feature - centroid) ** 2, axis=1)
                select_index = np.argmin(distance)
            else:
                print("The represent data select strategy is not given")
                exit()
            filtered_datalist.append(data_list[label_list[select_index]])
    print("After filter we choose {} data as our dataset".format(len(filtered_datalist)))
    return filtered_datalist


if __name__ == "__main__":
    import __init__
    from visual_feature import visual_multi_label_feature
    import os
    import torch
    from os import path
    from tensorboardX import SummaryWriter
    import re

    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    resume_path = "/data/fhz/unsupervised_feature_extractor_train_time_2/model_best.pth.tar"
    if os.path.isfile(resume_path):
        print("=> loading the resumed file '{}'".format(path.basename(resume_path)))
        checkpoint = torch.load(resume_path)
        train_feature = checkpoint['train_feature']
        test_feature = checkpoint['test_feature']
        train_datalist = checkpoint['train_datalist']
        test_datalist = checkpoint['test_datalist']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(path.basename(resume_path), checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume_path))
        exit()
    writer = SummaryWriter(log_dir="visual_time_2", comment="visual_features_with_spectral_label")
    num_data_list = [100, 200, 300, 400, 500, 600, 700, 800, 900]
    affinity_matrix = prob_sim_matrix(train_feature)
    for idx, num_data in enumerate(num_data_list):
        cls_label = [int(eval(re.match("(.*)_(.*)_annotations.npy", path.basename(raw_cube_path)).group(2)) > 3)
                     for raw_cube_path in train_datalist]
        cluster_label = list(SpectralClustering(n_clusters=num_data, affinity="precomputed", n_init=50, n_jobs=4).fit(
            affinity_matrix).labels_)
        label = list(zip(cls_label, cluster_label))
        label_header = ["classification", "cluster"]
        visual_multi_label_feature(feature=train_feature, label=label, label_header=label_header,
                                   data_list=train_datalist, writer=writer, global_step=(idx + 1),
                                   tag="numdata_{}".format(num_data))
