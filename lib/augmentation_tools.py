from skimage.transform import AffineTransform, rotate, PiecewiseAffineTransform, resize, warp
import numpy as np
import math
from numpy import random


class AugMethod2D(object):
    def __init__(self, image):
        """
        :param image: image should be a 2D picture with size(height,width,channels)
        or if image is a gray image,then it can also be size(height,width), the length of which is 2
        """
        self.image = image

    def flip(self, trans_position):
        if trans_position == "lr":
            self.image = np.fliplr(self.image)
        if trans_position == "ud":
            self.image = np.flipud(self.image)

    def rotate(self, angle):
        image_rotate = rotate(self.image, angle, resize=True, mode="reflect")
        if image_rotate.shape != self.image.shape:
            rescale_factor = (image_rotate.shape[0] / self.image.shape[0] +
                              image_rotate.shape[1] / self.image.shape[1]) / 2
            if rescale_factor > 1:
                rescale_factor = 1 / rescale_factor
            image_rotate = resize(image_rotate, self.image.shape, mode="reflect", anti_aliasing=True,
                                  anti_aliasing_sigma=None)
        self.image = image_rotate

    def aff_trans(self, x_cut=None, y_cut=None):
        if x_cut is None and y_cut is None:
            img_height, img_width, *_ = self.image.shape
            x_cut = -0.15 * img_height
            y_cut = -0.15 * img_width
        affine_trans_matrix = np.array([
            [1.2, 0.15, x_cut],
            [0.25, 1.2, y_cut],
            [0, 0, 1]])
        image_aff = warp(self.image, AffineTransform(matrix=affine_trans_matrix), mode="reflect")
        self.image = image_aff

    def add_noise(self, *args):
        """
        :param args : the args for the random sample function,args[0] is the distribution in numpy.random module
        :return: image with added noise
        """
        args = list(args)
        args.append(self.image.shape)
        noise = getattr(random, args[0])(*args[1:])
        self.image += noise

    @staticmethod
    def sin_aff_trans(image, rescale):
        return None


class AugMethod3D(object):
    def __init__(self, image):
        """
        :param image: image should be a 3D picture with size(depth,height,width,channels)
        or if image is a gray image,then it can also be size(depth,height,width), the length of which is 2
        or the image can also be a flat image s.t. size(k*height,k*width,channels) or (k*height,k*width)
        """
        self.image = image

    def flip(self, trans_position):
        """
        :param trans_position: can be one of item in ["D","H","W"], choose a trans position and then
        we will do flip in this position
        :return: the flip_lr image in location
        """
        if trans_position == "D":
            self.image = self.image[::-1, ...]
        if trans_position == "H":
            self.image = self.image[:, ::-1, ...]
        if trans_position == "W":
            self.image = self.image[:, :, ::-1, ...]

    def rotate(self, angle):
        """
        :param angle: do rotation for each slice and restore them into an image
        :return: image after rotation
        """
        image_rotated = np.empty(self.image.shape)
        for index, image_slice in enumerate(self.image):
            rotated_slice = rotate(image_slice, angle, resize=True, mode="reflect")
            if rotated_slice.shape != image_slice.shape:
                rescale_factor = (rotated_slice.shape[0] / image_slice.shape[0] +
                                  rotated_slice.shape[1] / image_slice.shape[1]) / 2
                if rescale_factor > 1:
                    rescale_factor = 1 / rescale_factor
                rotated_slice = resize(rotated_slice, image_slice.shape, mode="reflect",
                                       anti_aliasing=True, anti_aliasing_sigma=(1 - rescale_factor) / 2)
            image_rotated[index, ...] = rotated_slice
        self.image = image_rotated

    def aff_trans(self, x_cut=None, y_cut=None):
        if x_cut is None and y_cut is None:
            img_height, img_width, *_ = self.image.shape
            x_cut = -0.15 * img_height
            y_cut = -0.15 * img_width
        afftrans_matrix = np.array([
            [1.2, 0.15, x_cut],
            [0.25, 1.2, y_cut],
            [0, 0, 1]])
        image_affined = np.empty(self.image.shape)
        for index, image_slice in enumerate(self.image):
            image_affined[index, ...] = warp(image_slice, AffineTransform(matrix=afftrans_matrix), mode="reflect")
        self.image = image_affined

    def add_noise(self, *args):
        """
        :param distribution: support "normal","uniform"
        :param args : the args for the random sample function,args[0] is the distribution in numpy.random module
        :return: image with added noise
        """
        args = list(args)
        args.append(self.image.shape)
        noise = getattr(random, args[0])(*args[1:])
        self.image += noise

    def trans_flat_2_cube(self, h, w):
        """
        if image is a flat image stated in __init__,we can turn it into a cube s.t.(k,h,w)
        :return: cube with (d,h,w)
        """
        H, W, *_ = self.image.shape
        assert H % h == 0 and W % w == 0, print("can't split the flat image into slices,H/h or W/w is not integer")
        d = int((H * W) / (h * w))
        num_row, num_col = int(H / h), int(W / w)
        cube = np.empty((d, h, w, *_))
        for row_i in range(num_row):
            for col_j in range(num_col):
                cube[row_i * num_col + col_j, ...] = self.image[row_i * h:(row_i + 1) * h, col_j * w:(col_j + 1) * w,
                                                     ...]
        self.image = cube

    @staticmethod
    def trans_cube_2_flat(image):
        """
        :param image: a cube with (d,h,w,channels),we need to return a flat image containing them. the size of image
        can be i in row * j in col,i*j=d,and we will return a flat image with i the largest factor of d in range(sqrtd)
        :return:
        """
        d, h, w, *_ = image.shape
        num_row = [i for i in range(1, int(math.sqrt(d)) + 1) if d % i == 0][-1]
        num_col = int(d / num_row)
        flat_image = np.empty((h * num_row, w * num_col, *_))
        for row_i in range(num_row):
            for col_j in range(num_col):
                flat_image[row_i * h:(row_i + 1) * h, col_j * w:(col_j + 1) * w, ...] = image[
                    row_i * num_col + col_j, ...]
        return flat_image


if __name__ == "__main__":
    # import json
    # import cv2
    # from os import path, remove
    # from glob import glob
    #
    # dataset_path = "/home/ry-feng/kaggle_ndsb2017-master/resources/generated_traindata/pixel-wise-cube"
    # cube_info_path = path.join(dataset_path, "CubeInfo_Dict.json")
    # with open(cube_info_path, 'r') as f:
    #     cubedict = json.load(f)
    # cube_name_list = list(cubedict.keys())
    # augment_parameter_dict = {
    #     "flip_D": ["flip", "D"], "flip_H": ["flip", "H"], "flip_W": ["flip", "W"],
    #     "rotate_30": ["rotate", 30], "rotate_90": ["rotate", 90], "rotate_120": ["rotate", 120],
    #     "rotate_180": ["rotate", 180], "rotate_270": ["rotate", 270], "rotate_300": ["rotate", 300],
    #     "aff_trans": ["aff_trans"]
    # }
    # with open(path.join(dataset_path, "augment_parameter.json"), "w", encoding="utf8") as f:
    #     json.dump(augment_parameter_dict, f)
    # cube_path_list = [path.join(dataset_path, "".join([cube_name, ".png"])) for cube_name in cube_name_list]
    # label_path_list = [path.join(dataset_path, "".join([cube_name, "_label", ".png"])) for cube_name in cube_name_list]
    # all_img_path_list = glob(path.join(dataset_path, "*.png"))
    # for img_path in all_img_path_list:
    #     if img_path not in cube_path_list and img_path not in label_path_list:
    #         remove(img_path)
    # for cube_name in cube_name_list:
    #     """
    #     augmented image name rule:
    #     raw_img_name.png
    #     raw_img_name_label.png
    #     raw_img_name_augmentmethodname.png
    #     raw_img_name_augmentmethodname_label.png
    #     """
    #     cube_path = path.join(dataset_path, "".join([cube_name, ".png"]))
    #     label_name = "".join([cube_name, "_label"])
    #     label_path = path.join(dataset_path, "".join([label_name, ".png"]))
    #     cube_flat = cv2.imread(cube_path)
    #     label_flat = cv2.imread(label_path)
    #     cube_flat = cv2.cvtColor(cube_flat, cv2.COLOR_RGB2GRAY)
    #     label_flat = cv2.cvtColor(label_flat, cv2.COLOR_RGB2GRAY)
    #     cube_augment = AugMethod3D(cube_flat)
    #     cube_augment.trans_flat_2_cube(64, 64)
    #     label_augment = AugMethod3D(label_flat)
    #     label_augment.trans_flat_2_cube(64, 64)
    #     # Now all image is cube
    #     for key_name, value in augment_parameter_dict.items():
    #         augmented_cube = getattr(cube_augment, value[0])(*value[1:])
    #         augmented_label = getattr(label_augment, value[0])(*value[1:])
    #         augmented_cube_path = path.join(dataset_path, "".join([cube_name, "_", key_name, ".png"]))
    #         augmented_label_path = path.join(dataset_path, "".join([cube_name, "_", key_name, "_label.png"]))
    #         augmented_cube_flat = cube_augment.trans_cube_2_flat(augmented_cube)
    #         augmented_label_flat = label_augment.trans_cube_2_flat(augmented_label)
    #         cv2.imwrite(augmented_cube_path, augmented_cube_flat)
    #         cv2.imwrite(augmented_label_path, augmented_label_flat)
    import cv2
    import os
    from os import path
    from glob import glob

    basepath = "/home/feng/Suggest_Annotation_In_Object_Detection/Dataset/slide_flat_cube"

    train_img_path_list = glob(path.join(basepath, "train_cube_dset", "*_img.png"))
    val_img_path_list = glob(path.join(basepath, "val_cube_dset", "*_img.png"))
    test_img_path_list = glob(path.join(basepath, "test_cube_dset", "*_img.png"))
    for train_img_path in train_img_path_list:
        img_flat = cv2.imread(train_img_path)
        img_flat = cv2.cvtColor(img_flat, cv2.COLOR_RGB2GRAY)
