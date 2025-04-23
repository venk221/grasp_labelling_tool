import glob
import os
import csv

from imageio import imread
from skimage.transform import resize

from .item import *
from .grasp_dataset import GraspDataset
import torch
IMAGE_SHAPE = (1024, 1024)


# class JacquardDataset(GraspDataset):
#     def __init__(self, dataset_path, output_size=(300, 300), dep=True, rgb=False, start=0.0, end=1.0, augment=False):
#         super().__init__(output_size, dep, rgb, resizeMethod='scale')

#         # Store dataset path
#         self.__dataset_path = dataset_path

#         self.output_size = output_size

#         self.col_imgs = glob.glob(os.path.join(self.__dataset_path, '*', '*', '*_RGB.png'))
#         self.col_imgs.sort()
#         self.col_imgs = GraspDataset.split_list(self.col_imgs, start, end)
#         self.pos_grasps = [col_img.replace('RGB.png', 'grasps.txt') for col_img in self.col_imgs]
#         self.neg_grasps = []
#         self.dep_data = [col_img.replace('RGB.png', 'perfect_depth.tiff') for col_img in self.col_imgs]

class JacquardDataset(GraspDataset):
    def __init__(self, dataset_path, output_size=(300, 300), dep=True, rgb=False,
                 start=0.0, end=1.0, augment=False, max_samples=None): 
        super().__init__(output_size, dep, rgb, resizeMethod='scale')

        self.__dataset_path = dataset_path
        self.output_size = output_size

        self.col_imgs = glob.glob(os.path.join(self.__dataset_path, '*', '*', '*_RGB.png'))
        self.col_imgs.sort()
        self.col_imgs = GraspDataset.split_list(self.col_imgs, start, end)

        if max_samples is not None:
            self.col_imgs = self.col_imgs[:max_samples]  

        self.pos_grasps = [col_img.replace('RGB.png', 'grasps.txt') for col_img in self.col_imgs]
        self.neg_grasps = []
        self.dep_data = [col_img.replace('RGB.png', 'perfect_depth.tiff') for col_img in self.col_imgs]

    def __len__(self):
        return len(self.col_imgs)

    def get_raw_grasps(self, idx):
        grasps = []
        prev_centers = []
        with open(self.pos_grasps[idx]) as grasp_file:
            grasp_reader = csv.reader(grasp_file, delimiter=';')
            for row in grasp_reader:
                center = np.array([float(row[0]), float(row[1])])
                angle = float(row[2]) * np.pi / 180.0
                size = np.array([float(row[3]), float(row[4])])
                length_delta = np.array([np.cos(angle) * size[0], np.sin(angle) * size[0]]) / 2
                width_delta = np.array([np.cos(angle + np.pi / 2) * size[1], np.sin(angle + np.pi / 2) * size[1]]) / 2
                rect_pts = np.array([center - length_delta - width_delta, center + length_delta - width_delta, center + length_delta + width_delta, center - length_delta + width_delta])
                rect_pts = np.flip(rect_pts, 1)
                grasps.append(Rectangle(rect_pts, 1))
                grasps[-1].set_pts(grasps[-1].pts() * self.output_size / IMAGE_SHAPE[0])
                grasps[-1].set_pts(grasps[-1].pts().astype(int))
        return grasps


    def __get_rgb(self, idx):
        rgb_img = imread(self.col_imgs[idx])
        rgb_img = resize(rgb_img, self.output_size, preserve_range=True).astype(rgb_img.dtype)
        rgb_img = rgb_img.astype(np.float32) / 255.0
        rgb_img -= rgb_img.mean()
        rgb_img = rgb_img.transpose((2, 0, 1))
        return rgb_img

    def __get_dep(self, idx):
        dep_img = imread(self.dep_data[idx])
        dep_img = GraspDataset.normalize(dep_img)
        return resize(dep_img, self.output_size, preserve_range=True)

    def get_by_id(self, idx):
        return Item(self.__get_rgb(idx), self.__get_dep(idx), self.get_raw_grasps(idx))

    def get_by_filename(self, filename):
        colIdx = self.col_imgs.index(f'{filename}_RGB.png')
        return self[colIdx]

