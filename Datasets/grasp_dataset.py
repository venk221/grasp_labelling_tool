from torch.utils.data import Dataset
import torch
import numpy as np
from skimage.draw import polygon
from skimage.transform import resize

from .item import *


class GraspDataset(Dataset):
    def __init__(self, output_size=(300, 300), dep=True, rgb=True, resizeMethod='crop'):
        if resizeMethod not in ['crop', 'scale']:
            raise ValueError('Resize method should be either one of [crop, scale]')
        if not dep and not rgb:
            raise ValueError('At least one of [dep, rgb] should be True')

        self.__output_size = output_size
        self.__dep = dep
        self.__rgb = rgb
        self.__resizeMethod = resizeMethod

    # Splits a list based on percentage, such that the new list starts at 'start'% and ends at 'end'%
    @staticmethod
    def split_list(lst, start, end=1.0):
        total = len(lst)
        return lst[int(total * start):int(total * end)]

    @staticmethod
    def normalize(img):
        return np.clip((img - img.mean()), -1, 1)

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    # These should be overriden by base classes
    def __len__(self):
        raise NotImplementedError()

    def get_by_id(self, idx):
        raise NotImplementedError()

    def get_by_filename(self, filename):
        raise NotImplementedError()

    def get_raw_grasps(self, idx):
        raise NotImplementedError()

    # Will return the positive grasps cropped and translated to the correct output size
    def get_grasps(self, idx):
        # Get raw item
        item = self.get_raw_grasps(idx)

        # Filter out only positive grasps
        grasps = [grasp for grasp in item if grasp.score() == 1.0]

        return grasps

    def __getitem__(self, idx):
        ret = self.get_by_id(idx)

        x = None

        if self.__dep:
            x = GraspDataset.numpy_to_torch(ret.dep_img)
        if self.__rgb:
            if self.__dep:
                x = self.numpy_to_torch(np.concatenate((np.expand_dims(ret.dep_img, 0), ret.rgb_img), 0))
            else:
                x = GraspDataset.numpy_to_torch(ret.rgb_img)

        pos = np.zeros(self.__output_size)
        angle = np.zeros(self.__output_size)
        width = np.zeros(self.__output_size)

        # Crop and transform the grasps accordingly
        ret.grasp_arr = self.get_grasps(idx)
        for grasp in ret.grasp_arr:
            region = rect_shape_to_pts(grasp.center(), grasp.angle(), grasp.length() / 3, grasp.width(), grasp.score()).pts()
            rr, cc = polygon(region[:, 0], region[:, 1], self.__output_size)
            pos[rr, cc] = grasp.score()
            angle[rr, cc] = grasp.angle()
            width[rr, cc] = grasp.length()
        width = np.clip(width, 0.0, 150.0) / 150.0

        return x, (self.numpy_to_torch(pos), self.numpy_to_torch(np.cos(2*angle)), self.numpy_to_torch(np.sin(2*angle)),
                   self.numpy_to_torch(width)), idx, 0.0, 0.0

