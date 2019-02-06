import json
import re
from os import path
from random import sample
import cv2
import numpy as np
from numpy import random
from torch.utils import data
from glob import glob


class SGCCDataset(data.Dataset):
    def __init__(self, data_path_list):
        """
        :param type: the type folder
        :param tag: if we use large folder or small
        """
        self._data_path_list = data_path_list

    def __getitem__(self, index):
        data_path = self._data_path_list[index]
        image = cv2.imread(data_path, 0)
        image_name = path.basename(data_path)[:-4]
        image = np.expand_dims(image, 0)
        image = (image - 255) / 255
        # rescale image into [0,1]
        return image, index, image_name

    def __len__(self):
        return len(self._data_path_list)
