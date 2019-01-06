import torch
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from collections import defaultdict
from os import path
import pickle
import __init__
from glob import glob
import re
from utils.calculate_dist import gaussian_kl_calculation_vec
from utils.inference_result import get_inference_result


class Recommendation(object):
    def __init__(self, data_list, affinity_matrix, GPU_flag=False):
        """
        :param data_list: the list for data path, which is an 1-1 map to the row of affinity matrix
        :param affinity_matrix: the matrix storing the affinity, here affinity matrix represents for distance
        :param GPU_flag: flag for using gpu
        """
        self._data_list = data_list
        self._affinity_matrix = affinity_matrix
        self._filtered_data_list = list([])
        self._affinity_matrix_tensor = torch.from_numpy(affinity_matrix)
        self._knn_matrix = torch.zeros(self._affinity_matrix_tensor.size())
        if GPU_flag:
            self._affinity_matrix_tensor = self._affinity_matrix_tensor.cuda()
            self._knn_matrix = self._knn_matrix.cuda()

    def spectral_filter(self, target_num, similarity_graph, k=None, represent_strategy="min_sum"):
        if "knn" in similarity_graph:
            if k is None:
                origin_num = self._affinity_matrix.shape[0]
                k = round(target_num / origin_num) + 2
            yd, yi = self._affinity_matrix_tensor.topk(k, dim=1, largest=False, sorted=True)
            for i in range(yi.size(0)):
                self._knn_matrix[i, yi[i, ...]] = 1
            self._knn_matrix = self._knn_matrix.cpu().numpy()
            self._knn_matrix = self._knn_matrix + self._knn_matrix.T
            if similarity_graph == "mutual-knn":
                """mutual means only mutual part can be connected"""
                self._knn_matrix = np.array((self._knn_matrix == 2), dtype=int)
            else:
                self._knn_matrix = np.array((self._knn_matrix > 0), dtype=int)
            spectral_cls_result = SpectralClustering(n_clusters=target_num, affinity="precomputed", n_init=50,
                                                     n_jobs=4).fit(
                self._knn_matrix)
            cls_labels = spectral_cls_result.labels_
            cls_dict = defaultdict(list)
            for index, cls_label in enumerate(cls_labels):
                cls_dict[str(cls_label)].append(index)
            for cls_name, label_list in cls_dict.items():
                if len(label_list) == 1:
                    self._filtered_data_list.append(self._data_list[label_list[0]])
                else:
                    # choose one point most representative for the label
                    sub_affinity_matrix = self._affinity_matrix[label_list, :][:, label_list]
                    # Here we have 2 possible methods:
                    # 1. calculate f_i =sum_j d_ij and target_i = argmin f_i (make the sum distance min)
                    # 2. calculate g_i = max_j d_ij , and target_i = argmin g_i (make the max distance min
                    # 3. calculate the centroid point of data and find the nearest data
                    if represent_strategy == "min_sum":
                        f = np.sum(sub_affinity_matrix, axis=1)
                        select_index = np.argmin(f)
                    elif represent_strategy == "min_max":
                        g = np.max(sub_affinity_matrix, axis=1)
                        select_index = np.argmin(g)
                    else:
                        raise NotImplementedError("represent strategy {} not implemented".format(represent_strategy))
                    self._filtered_data_list.append(self._data_list[label_list[select_index]])
        else:
            raise NameError("Not found similarity graph :{}".format(similarity_graph))

    def __call__(self, *args, **kwargs):
        if kwargs["filter_name"] == "spectral":
            self.spectral_filter(target_num=kwargs["target_num"], similarity_graph=kwargs["similarity_graph"],
                                 k=kwargs["k"], represent_strategy=kwargs["represent_strategy"])
            return self._filtered_data_list
        else:
            raise NotImplementedError("filter {} not implemented".format(kwargs["filter_name"]))


def recommendation_vae(train_time, filter_name="spectral", target_num=100, similarity_graph="knn", k=5,
                       represent_strategy="min_sum"):
    train_label, mu_vec, log_sigma_vec, *_ = get_inference_result(train_time=train_time)
    data_list = [path.join("/data/fhz/lidc_cubes_64_overbound/npy", dname + ".npy") for dname in dname_list]
    kl = gaussian_kl_calculation_vec(mu_vec, log_sigma_vec, GPU_flag=True).cpu().numpy()
    recommendation = Recommendation(data_list, kl)
    recommendation_result = recommendation(filter_name=filter_name, target_num=target_num,
                                           similarity_graph=similarity_graph,
                                           k=k, represent_strategy=represent_strategy)
    return recommendation_result
