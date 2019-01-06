import json
import re
from os import path
from random import sample
import cv2
import numpy as np
from numpy import random
from torch.utils import data
from augmentation_tools import AugMethod3D, AugMethod2D
from utils.normalize_hu import cut_and_normalize_hu


class LungDataSet(data.Dataset):
    def __init__(self, data_path_list, need_name_label=False, need_malignancy_label=False,
                 need_seg_label=False, augment_prob=float(0), window_width=1500,
                 window_level=-400, cube_reshape=[32, 48, 48]):
        """
        :param data_path_list: the list store the whole data path,e.g.["/lungnodule1.npy","/lungnodule2.npy"]
        :param need_idx_label: True then we will return idx label
        :param need_name_label: True then we will return data name for each data
        :param need_malignancy_label: True then we will return malignancy label
        :param need_seg_label: True then we will return semantic segmentation label
        :param augment_prob: the probability of augmentation
        :param window_level: the window level of CT
        :param window_width: the window width of CT
        :param cube_shape : [depth, height, width]
        """
        self._data_path_list = data_path_list
        self._need_name_label = need_name_label
        self._need_malignancy_label = need_malignancy_label
        self._need_seg_label = need_seg_label
        self._augment_prob = augment_prob
        self._flip_arg = ["lr", "ud"]
        self._rotate_arg = [90, 180, 270]
        self._window_width = window_width
        self._window_level = window_level
        self._cube_reshape = cube_reshape

    def __getitem__(self, index):
        data_path = self._data_path_list[index]
        malignancy = "None"
        segmentation = "None"
        data_name = "None"
        image = "None"
        try:
            image = np.squeeze(np.load(data_path))
        except FileNotFoundError:
            print("can't load file :{}".format(data_path))
            input()
            exit(0)
        image = cut_and_normalize_hu(cube=image, window_width=self._window_width, window_level=self._window_level,
                                     cube_reshape=self._cube_reshape, augment_prob=self._augment_prob)
        if np.random.uniform(0, 1, 1)[0] < self._augment_prob:
            image = AugMethod3D(image)
            image.flip(trans_position=sample(self._flip_arg, 1))
            image.rotate(angle=sample(self._rotate_arg, 1))
            image.add_noise("uniform", -0.05, 0.05)
            image = image.image
        image = np.expand_dims(image, 0)
        if self._need_name_label:
            data_name = path.basename(data_path)[:-4]
        if self._need_malignancy_label:
            malignancy = eval(re.match("(.*)_(.*)_annotations.npy", path.basename(data_path)).group(2))
            if malignancy > 3:
                malignancy = 1
            else:
                malignancy = 0
        if self._need_seg_label:
            input("Here we don't have semantic segmentation label")
            exit()
        return image, index, data_name, malignancy, segmentation

    def __len__(self):
        return len(self._data_path_list)


class GlandDataset(data.Dataset):
    def __init__(self, data_path_list, need_seg_label=True, need_name_label=False, augment_prob=float(0)):
        self._data_path_list = data_path_list
        self._need_seg_flag = need_seg_label
        self._need_name_flag = need_name_label
        self._augment_prob = augment_prob
        self._flip_arg = ["lr", "ud"]
        self._rotate_arg = [90, 180, 270]

    def __getitem__(self, index):
        image_path = self._data_path_list[index]
        annotation_path = path.join(path.dirname(path.dirname(image_path)), "train_annotation",
                                    path.basename(image_path)[:-4] + "_anno.npy")
        image = np.load(image_path)
        # image is a (height,width,channel) array
        # with range [0,1]
        segmentation = "None"
        data_name = "None"
        if self._need_seg_flag:
            try:
                # segmentation is a 512*768 binary mask
                segmentation = np.load(annotation_path)
            except FileNotFoundError:
                print("The segmentation {} not found".format(image_path))
        if np.random.uniform(0, 1, 1)[0] < self._augment_prob:
            trans_position = sample(self._flip_arg, 1)
            angle = sample(self._rotate_arg, 1)
            image = AugMethod2D(image)
            image.flip(trans_position=trans_position)
            image.rotate(angle=angle)
            image.add_noise("uniform", -0.05, 0.05)
            image = image.image
            if self._need_seg_flag:
                segmentation = AugMethod2D(segmentation)
                segmentation.flip(trans_position=trans_position)
                segmentation.rotate(angle=angle)
                segmentation = segmentation.image
        image = np.transpose(image, [2, 0, 1])
        if self._need_name_flag:
            data_name = path.basename(self._data_path_list[index])[:-4]
        return image, index, data_name, segmentation

    def __len__(self):
        return len(self._data_path_list)
