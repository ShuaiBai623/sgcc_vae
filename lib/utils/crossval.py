"""
Here is the function for multi cross validation
"""
from os import path
from glob import glob
import numpy as np

dataset_path = "/data/fhz/lidc_cubes_64_overbound/npy"
datapath_list = glob(path.join(dataset_path, "*.npy"))


def multi_cross_validation(datapath_list=datapath_list, total_folder=5, target_folder=1):
    multi_cross_idx = list(np.linspace(0, len(datapath_list), total_folder + 1))
    multi_cross_idx = [int(round(i)) for i in multi_cross_idx]
    test = datapath_list[multi_cross_idx[target_folder - 1]:multi_cross_idx[target_folder]]
    train = [p for p in datapath_list if p not in test]
    return train, test
