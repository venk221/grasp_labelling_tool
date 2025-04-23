# Reads files from the cornell dataset
# Expects the cornell directory to be organized as follows:
# {CORNELL_FOLDER}:
#   \> 01
#   \> 02
#   \> 03

import glob
import os

import cv2
from imageio import imread

from .item import *
from .grasp_dataset import GraspDataset

# Important static constants
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
# Image has 480 rows, 640 columns
IMAGE_SHAPE = (480, 640)


class CornellDataset(GraspDataset):
    def __init__(self, dataset_path, output_size=(300, 300), dep=True, rgb=True, start=0.0, end=1.0):
        super().__init__(output_size, dep, rgb)
        # Store dataset path
        self.__dataset_path = dataset_path
        # Store the output size
        self.__output_size = output_size

        # Begin building the arrays of all required data (pos/neg rectangles, rgb/pcl data...)
        # Starting with color images:
        self.col_imgs = glob.glob(os.path.join(self.__dataset_path, '*', 'pcd*.png'))
        self.col_imgs.sort()
        self.col_imgs = GraspDataset.split_list(self.col_imgs, start, end)
        

        # Now create the others
        self.dep_imgs = [img.replace('r.png', 'd.tiff') for img in self.col_imgs]
        self.pos_rect = [img.replace('r.png', 'cpos.txt') for img in self.col_imgs]
        self.neg_rect = [img.replace('r.png', 'cneg.txt') for img in self.col_imgs]
        self.pcl_data = [img.replace('r.png', '.txt') for img in self.col_imgs]

    def __len__(self):
        return len(self.col_imgs)

    def crop(self, img, crop_params):
        # Get center of all grasps
        center, left, top = crop_params

        return img[top:min(img.shape[0], top + self.__output_size[0]), left:min(img.shape[1], left + self.__output_size[1])]

    def get_crop_params(self, grasps):
        means = np.zeros((len(grasps), 2))
        for idx, grasp in enumerate(grasps):
            means[idx] = np.mean(grasp.pts(), axis=0)

        center = np.flip(np.mean(means, axis=0, dtype=np.int64))
        top = max(0, min(center[0] - self.__output_size[0] // 2, IMAGE_SHAPE[0] - self.__output_size[0]))
        left = max(0, min(center[1] - self.__output_size[1] // 2, IMAGE_SHAPE[1] - self.__output_size[1]))

        return center, top, left

    @staticmethod
    def __inpaint(img):
        ret = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (ret == 0).astype(np.uint8)

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.abs(ret).max()
        ret = cv2.inpaint(ret.astype(np.float32) / scale, mask, 1, cv2.INPAINT_NS)

        # Back to original size and value range.
        ret = ret[1:-1, 1:-1] * scale

        return ret

    @staticmethod
    def __pcl_file_to_img(pcl_file, img_shape):
        ret = np.zeros(img_shape)
        with open(pcl_file) as pcl_data_file:
            for line in pcl_data_file.readlines():
                ls = line.split()

                # Some checks
                if len(ls) != 5:
                    # Not a point line in the file.
                    continue
                try:
                    # Not a number, carry on.
                    float(ls[0])
                except ValueError:
                    continue

                i = int(ls[4])
                r = i // img_shape[1]
                c = i % img_shape[1]
                x = float(ls[0])
                y = float(ls[1])
                z = float(ls[2])
                ret[r, c] = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        return ret / 1000

    def __pcl_to_depth(self, idx):
        ret = CornellDataset.__pcl_file_to_img(self.pcl_data[idx], IMAGE_SHAPE)
        # Before we return, let's inpaint since we can have missing points
        ret = CornellDataset.__inpaint(ret)
        return ret

    def __load_rgb(self, idx):
        rgb_img = imread(self.col_imgs[idx])
        crop_params = self.get_crop_params(CornellDataset.__load_grasp_file(self.pos_rect[idx], 1.0))
        rgb_img = self.crop(rgb_img, crop_params)
        rgb_img = np.resize(rgb_img, (self.__output_size[0], self.__output_size[1], 3))
        # rgb_img = np.resize(rgb_img, (self.__output_size[0], self.__output_size[1], 3))
        rgb_img = rgb_img.astype(np.float32) / 255.0
        rgb_img -= rgb_img.mean()
        rgb_img = rgb_img.transpose((2, 0, 1))
        return rgb_img

    def __load_dep(self, idx):
        # Crop based on the grasp array
        img = imread(self.dep_imgs[idx])
        crop_params = self.get_crop_params(CornellDataset.__load_grasp_file(self.pos_rect[idx], 1.0))
        img = self.crop(img, crop_params)
        img = np.resize(img, self.__output_size)
        return GraspDataset.normalize(img)

    @staticmethod
    def __load_grasp_file(grasp_file, score=0.0):
        ret = []
        with open(grasp_file) as gf:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = gf.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = gf.readline(), gf.readline(), gf.readline()
                # Convert to a 4x2 array of numeric values ((x, y), (x, y)...)
                try:
                    grasp_pts = np.array([[int(round(float(pt.split()[1]))), int(round(float(pt.split()[0])))]
                                          for pt in [p0, p1, p2, p3]])
                except ValueError:  # Failed to convert since there are NaNs in the data, thx cornell
                    continue
                ret.append(Rectangle(grasp_pts, score))
        return ret

    def __load_pos_grasps(self, idx):
        grasps = CornellDataset.__load_grasp_file(self.pos_rect[idx], 1.0)
        # Crop the grasps accordingly
        center, left, top = self.get_crop_params(grasps)

        for grasp in grasps:
            grasp.set_pts(grasp.pts() + np.array([-top, -left]).reshape((1, 2)))

        return grasps

    def __load_neg_grasps(self, idx):
        return CornellDataset.__load_grasp_file(self.neg_rect[idx], 0.0)

    def get_raw_grasps(self, idx):
        return self.__load_pos_grasps(idx) + self.__load_neg_grasps(idx)

    def get_by_id(self, idx):
        pos_grasps = self.__load_pos_grasps(idx)
        neg_grasps = self.__load_neg_grasps(idx)
        return Item(self.__load_rgb(idx), self.__load_dep(idx), pos_grasps) #+ neg_grasps

    # Expects filename made of four digits
    def get_by_filename(self, filename):
        for idx, col_img in enumerate(self.col_imgs):
            if f'pcd{filename}r.png' in col_img:
                return self[idx]
            



"""# import glob
# import os
# import math
# import cv2
# import numpy as np
# from imageio import imread
# from .item import *
# from .grasp_dataset import GraspDataset

# IMAGE_WIDTH = 640
# IMAGE_HEIGHT = 480
# IMAGE_SHAPE = (480, 640)


# class CornellDataset(GraspDataset):
#     def __init__(self, dataset_path, output_size=(300, 300), dep=True, rgb=True,
#                  start=0.0, end=1.0, augment=False):
#         super().__init__(output_size, dep, rgb)
#         self.__dataset_path = dataset_path
#         self.__output_size = output_size
#         self.__augment = augment

#         self.original_imgs = glob.glob(os.path.join(self.__dataset_path, '*', 'pcd*.png'))
#         self.original_imgs.sort()
#         self.original_imgs = GraspDataset.split_list(self.original_imgs, start, end)

#         self.orig_len = len(self.original_imgs)
#         if augment:
#             self.rotations = [0, 90, 180, 270]
#             self.zooms = [0.9, 1.0, 1.1]
#             self.total_aug_per_image = len(self.rotations) * len(self.zooms)
#             self.col_imgs = self.original_imgs * self.total_aug_per_image  # Logical expansion
#         else:
#             self.col_imgs = self.original_imgs
#             self.total_aug_per_image = 1

#         self.dep_imgs = [img.replace('r.png', 'd.tiff') for img in self.col_imgs]
#         self.pos_rect = [img.replace('r.png', 'cpos.txt') for img in self.col_imgs]
#         self.neg_rect = [img.replace('r.png', 'cneg.txt') for img in self.col_imgs]

#     def __len__(self):
#         return self.orig_len * self.total_aug_per_image if self.__augment else self.orig_len

#     def crop(self, img, crop_params):
#         center, left, top = crop_params
#         return img[top:min(img.shape[0], top + self.__output_size[0]),
#                    left:min(img.shape[1], left + self.__output_size[1])]

#     def get_crop_params(self, grasps):
#         means = np.zeros((len(grasps), 2))
#         for idx, grasp in enumerate(grasps):
#             means[idx] = np.mean(grasp.pts(), axis=0)
#         center = np.flip(np.mean(means, axis=0, dtype=np.int64))
#         top = max(0, min(center[0] - self.__output_size[0] // 2, IMAGE_SHAPE[0] - self.__output_size[0]))
#         left = max(0, min(center[1] - self.__output_size[1] // 2, IMAGE_SHAPE[1] - self.__output_size[1]))
#         return center, top, left

#     @staticmethod
#     def __inpaint(img):
#         ret = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
#         mask = (ret == 0).astype(np.uint8)
#         scale = np.abs(ret).max()
#         ret = cv2.inpaint(ret.astype(np.float32) / scale, mask, 1, cv2.INPAINT_NS)
#         return ret[1:-1, 1:-1] * scale

#     def __load_rgb(self, idx):
#         base_idx, rot, zoom = self.__get_aug_params(idx)
#         rgb_img = imread(self.original_imgs[base_idx])
#         crop_params = self.get_crop_params(self.__load_pos_grasps(base_idx))
#         rgb_img = self.crop(rgb_img, crop_params)
#         rgb_img = cv2.resize(rgb_img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
#         rgb_img = cv2.resize(rgb_img, self.__output_size)
#         rgb_img = rgb_img.astype(np.float32) / 255.0
#         rgb_img -= rgb_img.mean()
#         rgb_img = np.rot90(rgb_img, k=rot // 90) if rot else rgb_img
#         rgb_img = rgb_img.transpose((2, 0, 1))
#         return rgb_img

#     def __load_dep(self, idx):
#         base_idx, rot, zoom = self.__get_aug_params(idx)
#         dep_img = imread(self.original_imgs[base_idx].replace('r.png', 'd.tiff'))
#         crop_params = self.get_crop_params(self.__load_pos_grasps(base_idx))
#         dep_img = self.crop(dep_img, crop_params)
#         dep_img = self.__inpaint(dep_img)
#         dep_img = cv2.resize(dep_img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
#         dep_img = cv2.resize(dep_img, self.__output_size)
#         dep_img = np.rot90(dep_img, k=rot // 90) if rot else dep_img
#         return GraspDataset.normalize(dep_img)

#     @staticmethod
#     def __load_grasp_file(grasp_file, score=0.0):
#         ret = []
#         with open(grasp_file) as gf:
#             while True:
#                 p0 = gf.readline()
#                 if not p0:
#                     break
#                 p1, p2, p3 = gf.readline(), gf.readline(), gf.readline()
#                 try:
#                     grasp_pts = np.array([[int(round(float(pt.split()[1]))), int(round(float(pt.split()[0])))]
#                                           for pt in [p0, p1, p2, p3]])
#                 except ValueError:
#                     continue
#                 ret.append(Rectangle(grasp_pts, score))
#         return ret

#     def __load_pos_grasps(self, idx):
#         base_idx, rot, zoom = self.__get_aug_params(idx)
#         grasps = CornellDataset.__load_grasp_file(self.original_imgs[base_idx].replace('r.png', 'cpos.txt'), 1.0)
#         center, left, top = self.get_crop_params(grasps)
#         for grasp in grasps:
#             pts = grasp.pts()
#             pts = pts + np.array([-top, -left]).reshape((1, 2))
#             if rot:
#                 pts = self.__rotate_pts(pts, rot)
#             if zoom != 1.0:
#                 pts = self.__zoom_pts(pts, zoom)
#             grasp.set_pts(pts)
#         return grasps

#     def __load_neg_grasps(self, idx):
#         base_idx, _, _ = self.__get_aug_params(idx)
#         return CornellDataset.__load_grasp_file(self.original_imgs[base_idx].replace('r.png', 'cneg.txt'), 0.0)

#     def get_raw_grasps(self, idx):
#         return self.__load_pos_grasps(idx) + self.__load_neg_grasps(idx)

#     def get_by_id(self, idx):
#         pos_grasps = self.__load_pos_grasps(idx)
#         neg_grasps = self.__load_neg_grasps(idx)
#         return Item(self.__load_rgb(idx), self.__load_dep(idx), pos_grasps)

#     def get_by_filename(self, filename):
#         for idx, col_img in enumerate(self.col_imgs):
#             if f'pcd{filename}r.png' in col_img:
#                 return self[idx]

#     # === Aug Helper Logic ===
#     def __get_aug_params(self, idx):
#         if not self.__augment:
#             return idx, 0, 1.0
#         base_idx = idx // self.total_aug_per_image
#         aug_idx = idx % self.total_aug_per_image
#         rot = self.rotations[aug_idx // len(self.zooms)]
#         zoom = self.zooms[aug_idx % len(self.zooms)]
#         return base_idx, rot, zoom

#     def __rotate_pts(self, pts, angle_deg):
#         angle_rad = np.deg2rad(angle_deg)
#         center = np.array(self.__output_size) / 2
#         rot_mat = np.array([
#             [np.cos(angle_rad), -np.sin(angle_rad)],
#             [np.sin(angle_rad),  np.cos(angle_rad)]
#         ])
#         return np.dot(pts - center, rot_mat.T) + center

#     def __zoom_pts(self, pts, zoom):
#         center = np.array(self.__output_size) / 2
#         return (pts - center) * zoom + center





# Expects the cornell directory to be organized as follows:
# {CORNELL_FOLDER}:
#   \> 01
#   \> 02
#   \> 03

import glob
import os
import numpy as np
import cv2
from imageio import imread

from .item import *
from .grasp_dataset import GraspDataset

# Important static constants
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
# Image has 480 rows, 640 columns
IMAGE_SHAPE = (480, 640)


class CornellDataset(GraspDataset):
    def __init__(self, dataset_path, output_size=(300, 300), dep=True, rgb=True, start=0.0, end=1.0, augment=False, max_samples=None):
        super().__init__(output_size, dep, rgb)
        # Store dataset path
        self.__dataset_path = dataset_path
        # Store the output size
        self.__output_size = output_size
        # Augmentation flag
        self.augment = augment

        # Begin building the arrays of all required data (pos/neg rectangles, rgb/pcl data...)
        # Starting with color images:
        self.col_imgs = glob.glob(os.path.join(self.__dataset_path, '*', 'pcd*.png'))
        self.col_imgs.sort()
        self.col_imgs = GraspDataset.split_list(self.col_imgs, start, end)

        if max_samples is not None:
            self.col_imgs = self.col_imgs[:max_samples]
        
        # If augmentation is enabled, multiply the dataset by 12 unique transformations
        if self.augment:
            self.original_len = len(self.col_imgs)
            self.col_imgs = self.col_imgs * 12  # 12 unique augmentations per image

        # Now create the others
        self.dep_imgs = [img.replace('r.png', 'd.tiff') for img in self.col_imgs]
        self.pos_rect = [img.replace('r.png', 'cpos.txt') for img in self.col_imgs]
        self.neg_rect = [img.replace('r.png', 'cneg.txt') for img in self.col_imgs]
        self.pcl_data = [img.replace('r.png', '.txt') for img in self.col_imgs]

    def __len__(self):
        return len(self.col_imgs)

    def crop(self, img, crop_params):
        center, left, top = crop_params
        return img[top:min(img.shape[0], top + self.__output_size[0]), left:min(img.shape[1], left + self.__output_size[1])]

    def get_crop_params(self, grasps):
        means = np.zeros((len(grasps), 2))
        for idx, grasp in enumerate(grasps):
            means[idx] = np.mean(grasp.pts(), axis=0)

        center = np.flip(np.mean(means, axis=0, dtype=np.int64))
        top = max(0, min(center[0] - self.__output_size[0] // 2, IMAGE_SHAPE[0] - self.__output_size[0]))
        left = max(0, min(center[1] - self.__output_size[1] // 2, IMAGE_SHAPE[1] - self.__output_size[1]))

        return center, top, left

    @staticmethod
    def __inpaint(img):
        ret = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (ret == 0).astype(np.uint8)
        ret = cv2.inpaint(ret.astype(np.float32) / np.abs(ret).max(), mask, 1, cv2.INPAINT_NS)
        ret = ret[1:-1, 1:-1] * np.abs(ret).max()
        return ret

    @staticmethod
    def __pcl_file_to_img(pcl_file, img_shape):
        ret = np.zeros(img_shape)
        with open(pcl_file) as pcl_data_file:
            for line in pcl_data_file.readlines():
                ls = line.split()
                if len(ls) != 5:
                    continue
                try:
                    float(ls[0])
                except ValueError:
                    continue
                i = int(ls[4])
                r = i // img_shape[1]
                c = i % img_shape[1]
                x, y, z = float(ls[0]), float(ls[1]), float(ls[2])
                ret[r, c] = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return ret / 1000

    def __pcl_to_depth(self, idx):
        ret = CornellDataset.__pcl_file_to_img(self.pcl_data[idx], IMAGE_SHAPE)
        ret = CornellDataset.__inpaint(ret)
        if self.augment:
            ret = self.__apply_augmentation(ret, idx)
        return ret

    def __apply_augmentation(self, img, idx):
        aug_idx = idx // self.original_len if self.augment else 0
        if not self.augment or aug_idx == 0:
            return img
        current_shape = self.__output_size  # (300, 300)

        if aug_idx == 1:
            return np.rot90(img, 1)  # 90°
        elif aug_idx == 2:
            return np.rot90(img, 2)  # 180°
        elif aug_idx == 3:
            return np.rot90(img, 3)  # 270°
        elif aug_idx == 4:
            return img[50:-50, 50:-50]  # Crop 50 pixels from each side
        elif aug_idx == 5:  # Zoom in
            cropped = img[25:-25, 25:-25]  # 250x250
            return cv2.resize(cropped, self.__output_size[::-1])  # Resize to 300x300
        elif aug_idx == 6:  # Zoom out
            zoomed_shape = (int(current_shape[0] * 1.2), int(current_shape[1] * 1.2))  # 360x360
            zoomed = cv2.resize(img, zoomed_shape[::-1])
            crop_top = (zoomed_shape[0] - current_shape[0]) // 2  # 30
            crop_left = (zoomed_shape[1] - current_shape[1]) // 2  # 30
            return zoomed[crop_top:crop_top + current_shape[0], crop_left:crop_left + current_shape[1]]
        elif aug_idx == 7:
            return np.flipud(img)  # Vertical flip
        elif aug_idx == 8:
            return np.fliplr(img)  # Horizontal flip
        elif aug_idx == 9:
            return np.rot90(np.flipud(img), 1)  # Vertical flip + 90°
        elif aug_idx == 10:
            return np.rot90(np.fliplr(img), 1)  # Horizontal flip + 90°
        elif aug_idx == 11:
            return img[75:-75, 75:-75]  # Different crop
        return img

    def __load_rgb(self, idx):
        rgb_img = imread(self.col_imgs[idx])
        crop_params = self.get_crop_params(CornellDataset.__load_grasp_file(self.pos_rect[idx], 1.0))
        rgb_img = self.crop(rgb_img, crop_params)
        if self.augment:
            rgb_img = self.__apply_augmentation(rgb_img, idx)
        rgb_img = np.resize(rgb_img, (self.__output_size[0], self.__output_size[1], 3))
        rgb_img = rgb_img.astype(np.float32) / 255.0
        rgb_img -= rgb_img.mean()
        rgb_img = rgb_img.transpose((2, 0, 1))
        return rgb_img

    def __load_dep(self, idx):
        img = imread(self.dep_imgs[idx])
        crop_params = self.get_crop_params(CornellDataset.__load_grasp_file(self.pos_rect[idx], 1.0))
        img = self.crop(img, crop_params)
        if self.augment:
            img = self.__apply_augmentation(img, idx)
        img = np.resize(img, self.__output_size)
        return GraspDataset.normalize(img)

    @staticmethod
    def __load_grasp_file(grasp_file, score=0.0):
        ret = []
        with open(grasp_file) as gf:
            while True:
                p0 = gf.readline()
                if not p0:
                    break
                p1, p2, p3 = gf.readline(), gf.readline(), gf.readline()
                try:
                    grasp_pts = np.array([[int(round(float(pt.split()[1]))), int(round(float(pt.split()[0])))]
                                          for pt in [p0, p1, p2, p3]])
                except ValueError:
                    continue
                ret.append(Rectangle(grasp_pts, score))
        return ret

    def __augment_grasps(self, grasps, idx):
        aug_idx = idx // self.original_len if self.augment else 0
        if not self.augment or aug_idx == 0:
            return grasps

        # Start with the cropped image size (300x300)
        current_shape = self.__output_size  # (300, 300)

        for grasp in grasps:
            pts = grasp.pts().astype(np.float64)
            if aug_idx == 1:  # 90°
                pts = np.array([[-y, x] for x, y in pts]) + [current_shape[0], 0]
            elif aug_idx == 2:  # 180°
                pts = np.array([[-x, -y] for x, y in pts]) + current_shape
            elif aug_idx == 3:  # 270°
                pts = np.array([[y, -x] for x, y in pts]) + [0, current_shape[1]]
            elif aug_idx == 4:  # Crop 50
                pts -= 50
                current_shape = (current_shape[0] - 100, current_shape[1] - 100)  # 200x200
            elif aug_idx == 5:  # Zoom in
                pts = (pts - 25) * (current_shape[0] / (current_shape[0] - 50))
                current_shape = self.__output_size  # Resized back to 300x300
            elif aug_idx == 6:  # Zoom out
                zoomed_shape = (int(current_shape[0] * 1.2), int(current_shape[1] * 1.2))
                pts = pts * (zoomed_shape[0] / current_shape[0]) - [60, 80]
                current_shape = self.__output_size  # Resized back to 300x300
            elif aug_idx == 7:  # Vertical flip
                pts[:, 0] = current_shape[0] - pts[:, 0]
            elif aug_idx == 8:  # Horizontal flip
                pts[:, 1] = current_shape[1] - pts[:, 1]
            elif aug_idx == 9:  # Vertical flip + 90°
                pts = np.array([[-y, x] for x, y in pts]) + [current_shape[0], 0]
                pts[:, 0] = current_shape[0] - pts[:, 0]
            elif aug_idx == 10:  # Horizontal flip + 90°
                pts = np.array([[-y, x] for x, y in pts]) + [current_shape[0], 0]
                pts[:, 1] = current_shape[1] - pts[:, 1]
            elif aug_idx == 11:  # Different crop
                pts -= 75
                current_shape = (current_shape[0] - 150, current_shape[1] - 150)  # 150x150

            # Scale the grasp points to match the final output_size (300x300)
            if current_shape != self.__output_size:
                print(f"Scaling grasp points from {current_shape} to {self.__output_size}")
                scale_x = self.__output_size[1] / current_shape[1]
                scale_y = self.__output_size[0] / current_shape[0]
                pts[:, 0] *= scale_y  # y-coordinate
                pts[:, 1] *= scale_x  # x-coordinate

            # Ensure grasp points are within bounds
            pts[:, 0] = np.clip(pts[:, 0], 0, self.__output_size[0] - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, self.__output_size[1] - 1)

            grasp.set_pts(pts)
        return grasps

    def __load_pos_grasps(self, idx):
        grasps = CornellDataset.__load_grasp_file(self.pos_rect[idx], 1.0)
        center, left, top = self.get_crop_params(grasps)
        for grasp in grasps:
            grasp.set_pts(grasp.pts() + np.array([-top, -left]).reshape((1, 2)))
        if self.augment:
            grasps = self.__augment_grasps(grasps, idx)
        return grasps

    def __load_neg_grasps(self, idx):
        grasps = CornellDataset.__load_grasp_file(self.neg_rect[idx], 0.0)
        if self.augment:
            grasps = self.__augment_grasps(grasps, idx)
        return grasps

    def get_raw_grasps(self, idx):
        return self.__load_pos_grasps(idx) + self.__load_neg_grasps(idx)

    def get_by_id(self, idx):
        pos_grasps = self.__load_pos_grasps(idx)
        neg_grasps = self.__load_neg_grasps(idx)
        return Item(self.__load_rgb(idx), self.__load_dep(idx), pos_grasps)

    def get_by_filename(self, filename):
        for idx, col_img in enumerate(self.col_imgs):
            if f'pcd{filename}r.png' in col_img:
                return self[idx]

"""



































# # Expects the cornell directory to be organized as follows:
# # {CORNELL_FOLDER}:
# #   \> 01
# #   \> 02
# #   \> 03

# import glob
# import os
# import numpy as np
# import cv2
# from imageio import imread

# from .item import *
# from .grasp_dataset import GraspDataset

# # Important static constants
# IMAGE_WIDTH = 640
# IMAGE_HEIGHT = 480
# # Image has 480 rows, 640 columns
# IMAGE_SHAPE = (480, 640)


# class CornellDataset(GraspDataset):
#     def __init__(self, dataset_path, output_size=(300, 300), dep=True, rgb=True, start=0.0, end=1.0, augment=False, max_samples=None):
#         super().__init__(output_size, dep, rgb)
#         # Store dataset path
#         self.__dataset_path = dataset_path
#         # Store the output size
#         self.__output_size = output_size
#         # Augmentation flag
#         self.augment = augment

#         # Begin building the arrays of all required data (pos/neg rectangles, rgb/pcl data...)
#         # Starting with color images:
#         self.col_imgs = glob.glob(os.path.join(self.__dataset_path, '*', 'pcd*.png'))
#         self.col_imgs.sort()
#         self.col_imgs = GraspDataset.split_list(self.col_imgs, start, end)

#         if max_samples is not None:
#             self.col_imgs = self.col_imgs[:max_samples]
        
#         # If augmentation is enabled, multiply the dataset by 12 unique transformations
#         if self.augment:
#             self.original_len = len(self.col_imgs)
#             self.col_imgs = self.col_imgs * 12  # 12 unique augmentations per image

#         # Now create the others
#         self.dep_imgs = [img.replace('r.png', 'd.tiff') for img in self.col_imgs]
#         self.pos_rect = [img.replace('r.png', 'cpos.txt') for img in self.col_imgs]
#         self.neg_rect = [img.replace('r.png', 'cneg.txt') for img in self.col_imgs]
#         self.pcl_data = [img.replace('r.png', '.txt') for img in self.col_imgs]

#     def __len__(self):
#         return len(self.col_imgs)

#     def crop(self, img, crop_params):
#         center, left, top = crop_params
#         return img[top:min(img.shape[0], top + self.__output_size[0]), left:min(img.shape[1], left + self.__output_size[1])]

#     def get_crop_params(self, grasps):
#         means = np.zeros((len(grasps), 2))
#         for idx, grasp in enumerate(grasps):
#             means[idx] = np.mean(grasp.pts(), axis=0)

#         center = np.flip(np.mean(means, axis=0, dtype=np.int64))
#         top = max(0, min(center[0] - self.__output_size[0] // 2, IMAGE_SHAPE[0] - self.__output_size[0]))
#         left = max(0, min(center[1] - self.__output_size[1] // 2, IMAGE_SHAPE[1] - self.__output_size[1]))

#         return center, top, left

#     @staticmethod
#     def __inpaint(img):
#         ret = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
#         mask = (ret == 0).astype(np.uint8)
#         ret = cv2.inpaint(ret.astype(np.float32) / np.abs(ret).max(), mask, 1, cv2.INPAINT_NS)
#         ret = ret[1:-1, 1:-1] * np.abs(ret).max()
#         return ret

#     @staticmethod
#     def __pcl_file_to_img(pcl_file, img_shape):
#         ret = np.zeros(img_shape)
#         with open(pcl_file) as pcl_data_file:
#             for line in pcl_data_file.readlines():
#                 ls = line.split()
#                 if len(ls) != 5:
#                     continue
#                 try:
#                     float(ls[0])
#                 except ValueError:
#                     continue
#                 i = int(ls[4])
#                 r = i // img_shape[1]
#                 c = i % img_shape[1]
#                 x, y, z = float(ls[0]), float(ls[1]), float(ls[2])
#                 ret[r, c] = np.sqrt(x ** 2 + y ** 2 + z ** 2)
#         return ret / 1000

#     def __pcl_to_depth(self, idx):
#         ret = CornellDataset.__pcl_file_to_img(self.pcl_data[idx], IMAGE_SHAPE)
#         ret = CornellDataset.__inpaint(ret)
#         if self.augment:
#             ret = self.__apply_augmentation(ret, idx)
#         return ret

#     def __apply_augmentation(self, img, idx):
#         aug_idx = idx // self.original_len if self.augment else 0
#         if not self.augment or aug_idx == 0:
#             return img
#         current_shape = self.__output_size  # (300, 300)

#         if aug_idx == 1:
#             return np.rot90(img, 1)  # 90° counterclockwise
#         elif aug_idx == 2:
#             return np.rot90(img, 2)  # 180°
#         elif aug_idx == 3:
#             return np.rot90(img, 3)  # 270° counterclockwise
#         elif aug_idx == 4:
#             return img[50:-50, 50:-50]  # Crop 50 pixels from each side
#         elif aug_idx == 5:  # Zoom in
#             cropped = img[25:-25, 25:-25]  # 250x250
#             return cv2.resize(cropped, self.__output_size[::-1])  # Resize to 300x300
#         elif aug_idx == 6:  # Zoom out
#             zoomed_shape = (int(current_shape[0] * 1.2), int(current_shape[1] * 1.2))  # 360x360
#             zoomed = cv2.resize(img, zoomed_shape[::-1])
#             crop_top = (zoomed_shape[0] - current_shape[0]) // 2  # 30
#             crop_left = (zoomed_shape[1] - current_shape[1]) // 2  # 30
#             return zoomed[crop_top:crop_top + current_shape[0], crop_left:crop_left + current_shape[1]]
#         elif aug_idx == 7:
#             return np.flipud(img)  # Vertical flip
#         elif aug_idx == 8:
#             return np.fliplr(img)  # Horizontal flip
#         elif aug_idx == 9:
#             return np.rot90(np.flipud(img), 1)  # Vertical flip + 90°
#         elif aug_idx == 10:
#             return np.rot90(np.fliplr(img), 1)  # Horizontal flip + 90°
#         elif aug_idx == 11:
#             return img[75:-75, 75:-75]  # Different crop
#         return img

#     def __load_rgb(self, idx):
#         rgb_img = imread(self.col_imgs[idx])
#         crop_params = self.get_crop_params(CornellDataset.__load_grasp_file(self.pos_rect[idx], 1.0))
#         rgb_img = self.crop(rgb_img, crop_params)
#         if self.augment:
#             rgb_img = self.__apply_augmentation(rgb_img, idx)
#         if rgb_img.shape[:2] != self.__output_size:
#             rgb_img = cv2.resize(rgb_img, self.__output_size[::-1], interpolation=cv2.INTER_LINEAR)
#         rgb_img = rgb_img.astype(np.float32) / 255.0
#         rgb_img -= rgb_img.mean()
#         rgb_img = rgb_img.transpose((2, 0, 1))
#         return rgb_img

#     def __load_dep(self, idx):
#         img = imread(self.dep_imgs[idx])
#         crop_params = self.get_crop_params(CornellDataset.__load_grasp_file(self.pos_rect[idx], 1.0))
#         img = self.crop(img, crop_params)
#         if self.augment:
#             img = self.__apply_augmentation(img, idx)
#         if img.shape[:2] != self.__output_size:
#             img = cv2.resize(img, self.__output_size[::-1], interpolation=cv2.INTER_LINEAR)
#         return GraspDataset.normalize(img)

#     @staticmethod
#     def __load_grasp_file(grasp_file, score=0.0):
#         ret = []
#         with open(grasp_file) as gf:
#             while True:
#                 p0 = gf.readline()
#                 if not p0:
#                     break
#                 p1, p2, p3 = gf.readline(), gf.readline(), gf.readline()
#                 try:
#                     grasp_pts = np.array([[int(round(float(pt.split()[1]))), int(round(float(pt.split()[0])))]
#                                           for pt in [p0, p1, p2, p3]])
#                 except ValueError:
#                     continue
#                 ret.append(Rectangle(grasp_pts, score))
#         return ret

#     def __augment_grasps(self, grasps, idx):
#         aug_idx = idx // self.original_len if self.augment else 0
#         if not self.augment or aug_idx == 0:
#             return grasps

#         # Start with the cropped image size (300x300), ensure float
#         current_shape = (float(self.__output_size[0]), float(self.__output_size[1]))  # (300.0, 300.0)
#         H, W = current_shape  # Height and Width (300, 300)

#         # Update current_shape once based on the augmentation
#         if aug_idx == 4:  # Crop 50
#             new_height = max(1.0, current_shape[0] - 100.0)  # Prevent zero
#             new_width = max(1.0, current_shape[1] - 100.0)
#             current_shape = (new_height, new_width)  # 200.0x200.0
#         elif aug_idx == 5:  # Zoom in
#             current_shape = (float(self.__output_size[0]), float(self.__output_size[1]))
#         elif aug_idx == 6:  # Zoom out
#             current_shape = (float(self.__output_size[0]), float(self.__output_size[1]))
#         elif aug_idx == 11:  # Different crop
#             new_height = max(1.0, current_shape[0] - 150.0)  # Prevent zero
#             new_width = max(1.0, current_shape[1] - 150.0)
#             current_shape = (new_height, new_width)  # 150.0x150.0

#         for grasp in grasps:
#             pts = grasp.pts().astype(np.float64)  # Convert to float64

#             # Scale the grasp points to match the final output_size (300x300) *before* rotation
#             if current_shape != (float(self.__output_size[0]), float(self.__output_size[1])):
#                 scale_x = self.__output_size[1] / max(1.0, current_shape[1])  # Prevent zero division
#                 scale_y = self.__output_size[0] / max(1.0, current_shape[0])
#                 pts[:, 0] *= scale_y  # y-coordinate
#                 pts[:, 1] *= scale_x  # x-coordinate
#                 current_shape = (float(self.__output_size[0]), float(self.__output_size[1]))  # Update current_shape

#             # Center the points for rotation (shift to origin)
#             center = np.array([W / 2, H / 2])
#             pts -= center

#             # Apply the augmentation transformations
#             if aug_idx == 1:  # 90° counterclockwise
#                 # Rotate around center: (x, y) -> (-y, x)
#                 pts = np.array([[-y, x] for x, y in pts])
#                 # Reorder points: [bottom-left, top-left, top-right, bottom-right] -> [top-left, top-right, bottom-right, bottom-left]
#                 pts = np.roll(pts, -1, axis=0)
#             elif aug_idx == 2:  # 180°
#                 # Rotate around center: (x, y) -> (-x, -y)
#                 pts = np.array([[-x, -y] for x, y in pts])
#                 # Reorder points: [bottom-right, bottom-left, top-left, top-right] -> [top-left, top-right, bottom-right, bottom-left]
#                 pts = np.roll(pts, -2, axis=0)
#             elif aug_idx == 3:  # 270° counterclockwise (90° clockwise)
#                 # Rotate around center: (x, y) -> (y, -x)
#                 pts = np.array([[y, -x] for x, y in pts])
#                 # Reorder points: [top-right, bottom-right, bottom-left, top-left] -> [top-left, top-right, bottom-right, bottom-left]
#                 pts = np.roll(pts, -3, axis=0)
#             elif aug_idx == 4:  # Crop 50
#                 pts += center  # Undo centering
#                 pts -= 50
#                 pts -= center  # Re-center for any further transformations
#             elif aug_idx == 5:  # Zoom in
#                 pts += center  # Undo centering
#                 pts = (pts - 25) * (self.__output_size[0] / max(1.0, (self.__output_size[0] - 50)))  # Prevent zero division
#                 pts -= center
#             elif aug_idx == 6:  # Zoom out
#                 pts += center  # Undo centering
#                 zoomed_shape = (int(self.__output_size[0] * 1.2), int(self.__output_size[1] * 1.2))
#                 pts = pts * (zoomed_shape[0] / max(1.0, self.__output_size[0])) - [60, 80]  # Prevent zero division
#                 pts -= center
#             elif aug_idx == 7:  # Vertical flip
#                 # (x, y) -> (x, -y)
#                 pts[:, 0] = -pts[:, 0]
#             elif aug_idx == 8:  # Horizontal flip
#                 # (x, y) -> (-x, y)
#                 pts[:, 1] = -pts[:, 1]
#             elif aug_idx == 9:  # Vertical flip + 90°
#                 # First, vertical flip: (x, y) -> (x, -y)
#                 pts[:, 0] = -pts[:, 0]
#                 # Then, 90°: (x, -y) -> (y, x)
#                 pts = np.array([[y, x] for x, y in pts])
#                 # Reorder points: [top-right, bottom-right, bottom-left, top-left] -> [top-left, top-right, bottom-right, bottom-left]
#                 pts = np.roll(pts, -3, axis=0)
#             elif aug_idx == 10:  # Horizontal flip + 90°
#                 # First, horizontal flip: (x, y) -> (-x, y)
#                 pts[:, 1] = -pts[:, 1]
#                 # Then, 90°: (-x, y) -> (y, x)
#                 pts = np.array([[y, x] for x, y in pts])
#                 # Reorder points: [bottom-left, top-left, top-right, bottom-right] -> [top-left, top-right, bottom-right, bottom-left]
#                 pts = np.roll(pts, -1, axis=0)
#             elif aug_idx == 11:  # Different crop
#                 pts += center  # Undo centering
#                 pts -= 75
#                 pts -= center

#             # Shift points back after rotation
#             pts += center

#             # Ensure grasp points are within bounds
#             pts[:, 0] = np.clip(pts[:, 0], 0, self.__output_size[0] - 1)
#             pts[:, 1] = np.clip(pts[:, 1], 0, self.__output_size[1] - 1)

#             grasp.set_pts(pts)
#         return grasps

#     def __load_pos_grasps(self, idx):
#         grasps = CornellDataset.__load_grasp_file(self.pos_rect[idx], 1.0)
#         center, left, top = self.get_crop_params(grasps)
#         for grasp in grasps:
#             grasp.set_pts(grasp.pts() + np.array([-top, -left]).reshape((1, 2)))
#         if self.augment:
#             grasps = self.__augment_grasps(grasps, idx)  # Remove top and left
#         return grasps

#     def __load_neg_grasps(self, idx):
#         grasps = CornellDataset.__load_grasp_file(self.neg_rect[idx], 0.0)
#         if self.augment:
#             grasps = self.__augment_grasps(grasps, idx)  # Remove top and left
#         return grasps

#     def get_raw_grasps(self, idx):
#         return self.__load_pos_grasps(idx) + self.__load_neg_grasps(idx)

#     def get_by_id(self, idx):
#         pos_grasps = self.__load_pos_grasps(idx)
#         neg_grasps = self.__load_neg_grasps(idx)
#         return Item(self.__load_rgb(idx), self.__load_dep(idx), pos_grasps)

#     def get_by_filename(self, filename):
#         for idx, col_img in enumerate(self.col_imgs):
#             if f'pcd{filename}r.png' in col_img:
#                 return self[idx]
