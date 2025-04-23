# import glob
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
#         grasps = CornellDataset.__load_grasp_file(self.pos_rect[idx], 1.0)

#         # Crop based on original grasp
#         crop_params = self.get_crop_params(grasps)
#         rgb_img = self.crop(rgb_img, crop_params)
#         for grasp in grasps:
#             grasp.set_pts(grasp.pts() + np.array([-crop_params[2], -crop_params[1]]))  # top, left

#         # Apply augmentation to both image and grasps
#         if self.augment:
#             rgb_img = self.__apply_augmentation(rgb_img, idx)
#             grasps = self.__augment_grasps(grasps, idx)

#         if rgb_img.shape[:2] != self.__output_size:
#             rgb_img = cv2.resize(rgb_img, self.__output_size[::-1], interpolation=cv2.INTER_LINEAR)

#         rgb_img = rgb_img.astype(np.float32) / 255.0
#         rgb_img -= rgb_img.mean()
#         rgb_img = rgb_img.transpose((2, 0, 1))
#         return rgb_img


#     def __load_dep(self, idx):
#         img = imread(self.dep_imgs[idx])
#         grasps = CornellDataset.__load_grasp_file(self.pos_rect[idx], 1.0)

#         crop_params = self.get_crop_params(grasps)
#         img = self.crop(img, crop_params)
#         for grasp in grasps:
#             grasp.set_pts(grasp.pts() + np.array([-crop_params[2], -crop_params[1]]))  # top, left

#         if self.augment:
#             img = self.__apply_augmentation(img, idx)
#             grasps = self.__augment_grasps(grasps, idx)

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
#         crop_params = self.get_crop_params(grasps)
#         for grasp in grasps:
#             grasp.set_pts(grasp.pts() + np.array([-crop_params[2], -crop_params[1]]))  # top, left

#         if self.augment:
#             grasps = self.__augment_grasps(grasps, idx)

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




# Expects the cornell directory to be organized as follows:
# {CORNELL_FOLDER}:
#   \> 01
#   \> 02
#   \> 03

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



import glob
import os

import numpy as np
import cv2
from imageio import imread

from .item import *
from .grasp_dataset import GraspDataset

# Image has 480 rows, 640 columns in Cornell
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)


class CornellDataset(GraspDataset):
    def __init__(self, dataset_path, output_size=(300, 300), dep=True, rgb=True,
                 start=0.0, end=1.0, augment=False, max_samples=None):
        """
        Same signature as your original code. If `augment=False`, identical behavior.
        If `augment=True`, we replicate each sample 12×, each with a different transform.
        """
        super().__init__(output_size, dep, rgb)

        self.__dataset_path = dataset_path
        self.__output_size = output_size
        self.augment = augment

        # 1) Gather and sort all RGB files: "pcdXXXXr.png"
        self.col_imgs = glob.glob(os.path.join(self.__dataset_path, '*', 'pcd*[0-9]r.png'))
        self.col_imgs.sort()

        # 2) Split based on start & end fraction
        self.col_imgs = GraspDataset.split_list(self.col_imgs, start, end)

        # 3) If a max_samples limit is given, truncate
        if max_samples is not None:
            self.col_imgs = self.col_imgs[:max_samples]

        # Store how many we had before augmentation
        self.original_len = len(self.col_imgs)

        # 4) If augment=True, replicate dataset ×12
        if self.augment:
            self.col_imgs = self.col_imgs * 12  # 12 transforms

        # 5) Build corresponding depth images & grasp text files
        #    e.g. pcdXXXXr.png => pcdXXXXd.tiff, pcdXXXXcpos.txt, pcdXXXXcneg.txt, pcdXXXX.txt
        self.dep_imgs = [img.replace('r.png', 'd.tiff') for img in self.col_imgs]
        self.pos_rect = [img.replace('r.png', 'cpos.txt') for img in self.col_imgs]
        self.neg_rect = [img.replace('r.png', 'cneg.txt') for img in self.col_imgs]
        self.pcl_data = [img.replace('r.png', '.txt')   for img in self.col_imgs]

    def __len__(self):
        return len(self.col_imgs)

    # -----------------------------------------------------
    # Exactly the same "crop" + "get_crop_params" as before
    # -----------------------------------------------------
    def crop(self, img, crop_params):
        center, top, left = crop_params
        h, w = self.__output_size
        cropped = img[top:top + h, left:left + w]

        # Fix shape if too short (can happen near image boundaries)
        if cropped.shape[0] != h or cropped.shape[1] != w:
            fixed = np.zeros((h, w), dtype=cropped.dtype)
            fixed[:cropped.shape[0], :cropped.shape[1]] = cropped
            return fixed
        return cropped


    def get_crop_params(self, grasps):
        """
        Original Cornell approach: place the average of all corners in the center.
        Return (center, top, left).
        """
        means = np.zeros((len(grasps), 2))
        for i, g in enumerate(grasps):
            means[i] = np.mean(g.pts(), axis=0)  # average [row, col]
        # overall center
        center_rc = np.mean(means, axis=0).astype(np.int64)

        out_h, out_w = self.__output_size
        # clamp top, left
        top = max(0, min(center_rc[0] - out_h // 2, IMAGE_SHAPE[0] - out_h))
        left = max(0, min(center_rc[1] - out_w // 2, IMAGE_SHAPE[1] - out_w))
        return (center_rc, top, left)

    # -----------------------------------------
    # Same inpainting & PCL reading as before
    # -----------------------------------------
    @staticmethod
    def __inpaint(img):
        """
        Inpaint missing depth in float32 image.
        """
        border = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (border == 0).astype(np.uint8)

        mx = np.abs(border).max()
        if mx < 1e-6:
            return img.copy()

        normed = border.astype(np.float32) / mx
        inpainted = cv2.inpaint(normed, mask, 1, cv2.INPAINT_NS)
        inpainted = inpainted[1:-1, 1:-1] * mx
        return inpainted

    @staticmethod
    def __pcl_file_to_img(pcl_file, img_shape):
        """
        Convert PCL text => depth array. shape=img_shape
        """
        ret = np.zeros(img_shape, dtype=np.float32)
        with open(pcl_file) as pcl_data_file:
            for line in pcl_data_file:
                ls = line.split()
                if len(ls) != 5:
                    continue
                try:
                    x, y, z = float(ls[0]), float(ls[1]), float(ls[2])
                    i = int(ls[4])
                except ValueError:
                    continue
                r = i // img_shape[1]
                c = i % img_shape[1]
                dist = np.sqrt(x**2 + y**2 + z**2)
                ret[r, c] = dist
        ret /= 1000.0  # mm -> m
        return ret

    def __pcl_to_depth(self, idx):
        """
        Load from self.pcl_data[idx], inpaint, then do the same
        cropping/augmentation as the standard __load_dep does.
        If you prefer, you can call this inside __load_dep.
        """
        depth_img = self.__pcl_file_to_img(self.pcl_data[idx], IMAGE_SHAPE)
        depth_img = self.__inpaint(depth_img)
        # Then do the same center-crop and possible augment
        # For consistency with your original code, we won't override __load_dep here.
        return depth_img

    # --------------------------------------------
    # The standard "load RGB" and "load depth",
    # now with optional 12x augmentation
    # --------------------------------------------
    def __load_rgb(self, idx):
        """
        1) Load the full 480×640 r.png
        2) Load positive grasps to find center-crop offset
        3) Crop the image to (300×300)
        4) If augment=False, done. If augment=True, apply transform ID (0..11)
        5) Final shape => (3, 300, 300) float32
        """
        base_idx, transform_id = self.__map_index(idx)

        # 1) read
        rgb_img = imread(self.col_imgs[idx])  # shape ~ (480,640,3 or 4)
        if rgb_img.shape[-1] == 4:  # if RGBA, ignore alpha
            rgb_img = rgb_img[..., :3]

        # 2) load pos grasps to find center-crop
        pos_grasps_full = self.__load_grasp_file(self.pos_rect[idx], 1.0)
        crop_params = self.get_crop_params(pos_grasps_full)
        rgb_cropped = self.crop(rgb_img, crop_params)

        # shift the pos grasps accordingly so they reflect the cropped region
        _, top, left = crop_params
        for g in pos_grasps_full:
            pts = g.pts()
            pts[:, 0] -= top
            pts[:, 1] -= left
            g.set_pts(pts)

        # 3) If no augmentation, just resize, normalize
        if not self.augment:
            if rgb_cropped.shape[:2] != self.__output_size:
                rgb_cropped = cv2.resize(rgb_cropped,
                                         (self.__output_size[1], self.__output_size[0]),
                                         interpolation=cv2.INTER_LINEAR)
            rgb_cropped = rgb_cropped.astype(np.float32) / 255.0
            rgb_cropped -= rgb_cropped.mean()
            # shape => (3,300,300)
            # if rgb_cropped.shape[-1] == 3:
            #     rgb_cropped = np.transpose(rgb_cropped, (2, 0, 1))  # (H,W,3) -> (3,H,W)

            if rgb_cropped.shape == self.__output_size:
                # If it's grayscale (H,W), make it 3-channel RGB
                rgb_cropped = np.stack([rgb_cropped] * 3, axis=-1)

            if rgb_cropped.shape[-1] == 3 and rgb_cropped.ndim == 3:
                rgb_cropped = np.transpose(rgb_cropped, (2, 0, 1))
            return rgb_cropped

        # 4) If augment=True, apply transform
        aug_img = self.__apply_augmentation_rgb(rgb_cropped, pos_grasps_full,
                                                transform_id, base_idx)

        # `aug_img` returned as (300,300,3). Let's do final mean-sub, transpose
        aug_img = aug_img.astype(np.float32) / 255.0
        aug_img -= aug_img.mean()
        if aug_img.shape[-1] == 3:
            aug_img = np.transpose(aug_img, (2, 0, 1))
        return aug_img

    def __load_dep(self, idx):
        base_idx, transform_id = self.__map_index(idx)

        depth_img = imread(self.dep_imgs[idx])  # (480, 640)

        pos_grasps_full = self.__load_grasp_file(self.pos_rect[idx], 1.0)
        crop_params = self.get_crop_params(pos_grasps_full)
        dep_cropped = self.crop(depth_img, crop_params)

        if dep_cropped.ndim != 2:
            # print(f"[!] DEPTH FIX TRIGGERED @ idx {idx}: crop shape = {dep_cropped.shape}")
            fixed = np.zeros(self.__output_size, dtype=np.float32)
            if dep_cropped.ndim == 1 and dep_cropped.shape[0] == self.__output_size[1]:
                fixed[0, :] = dep_cropped
            dep_cropped = fixed

        if not self.augment:
            if dep_cropped.shape != self.__output_size:
                # print(f"[!] DEP BAD FINAL SHAPE @ idx {idx}: {dep_cropped.shape}")
                dep_cropped = cv2.resize(dep_cropped, self.__output_size[::-1], interpolation=cv2.INTER_NEAREST)
            return GraspDataset.normalize(dep_cropped)

        aug_img = self.__apply_augmentation_depth(dep_cropped, pos_grasps_full, transform_id, base_idx)

        if aug_img.shape != self.__output_size:
            # print(f"[!] DEPTH AUG SHAPE FIX @ idx {idx}: {aug_img.shape}")
            aug_img = cv2.resize(aug_img, self.__output_size[::-1], interpolation=cv2.INTER_NEAREST)

        return GraspDataset.normalize(aug_img)


    @staticmethod
    def __load_grasp_file(grasp_file, score=0.0):
        """
        Load 4 lines => corners, each line "x y".
        Return a list of Rectangles with .pts() as [row,col].
        """
        ret = []
        with open(grasp_file, 'r') as gf:
            while True:
                p0 = gf.readline()
                if not p0:
                    break
                p1, p2, p3 = gf.readline(), gf.readline(), gf.readline()
                try:
                    corners = np.array([
                        [int(round(float(pt.split()[1]))),  # row (y)
                         int(round(float(pt.split()[0])))]  # col (x)
                        for pt in [p0, p1, p2, p3]
                    ])
                except ValueError:
                    continue
                ret.append(Rectangle(corners, score=score))
        return ret

    # -----------------------------------------------------
    # The original "load_pos_grasps" and "load_neg_grasps"
    # now with optional augmentation
    # -----------------------------------------------------
    def __load_pos_grasps(self, idx):
        """
        Return the positive grasps in final (300×300) coords.
        If augment=False, it's just the center-crop shift.
        If augment=True, also apply transform ID to them.
        """
        base_idx, transform_id = self.__map_index(idx)
        grasps = self.__load_grasp_file(self.pos_rect[idx], 1.0)
        crop_params = self.get_crop_params(grasps)
        _, top, left = crop_params

        # shift them for the crop
        for g in grasps:
            pts = g.pts()
            pts[:, 0] -= top
            pts[:, 1] -= left
            g.set_pts(pts)

        # done if not augment
        if not self.augment:
            return grasps

        # else apply the same transform
        aug_grasps = self.__augment_grasp_list(grasps, transform_id, base_idx)
        return aug_grasps

    def __load_neg_grasps(self, idx):
        """
        Return the negative grasps in final (300×300) coords.
        If augment=True, also transform them.
        """
        base_idx, transform_id = self.__map_index(idx)
        grasps = self.__load_grasp_file(self.neg_rect[idx], 0.0)
        # use same top/left from pos so everything is consistent
        pos_grasps = self.__load_grasp_file(self.pos_rect[idx], 1.0)
        crop_params = self.get_crop_params(pos_grasps)
        _, top, left = crop_params

        for g in grasps:
            pts = g.pts()
            pts[:, 0] -= top
            pts[:, 1] -= left
            g.set_pts(pts)

        if not self.augment:
            return grasps

        aug_grasps = self.__augment_grasp_list(grasps, transform_id, base_idx)
        return aug_grasps

    def get_raw_grasps(self, idx):
        """
        Return *all* grasps (pos+neg) at final resolution (300×300).
        """
        return self.__load_pos_grasps(idx) + self.__load_neg_grasps(idx)

    def get_by_id(self, idx):
        """
        Return an Item with (rgb, depth, grasps).
        rgb => (3,300,300), depth => (300,300)
        """
        pos = self.__load_pos_grasps(idx)
        neg = self.__load_neg_grasps(idx)
        rgb = self.__load_rgb(idx)
        dep = self.__load_dep(idx)
        if rgb.shape != (3, 300, 300):
            print(f"[!] BAD RGB SHAPE @ idx {idx}: {rgb.shape}")
        if dep.shape != (300, 300):
            print(f"[SHAPE CHECK] RGB: {rgb.shape}, DEP: {dep.shape}, idx: {idx}")
        return Item(rgb, dep, pos ) #+ neg

    def get_by_filename(self, filename):
        """
        Search for 'pcd{filename}r.png' in self.col_imgs; return that item.
        """
        for i, cimg in enumerate(self.col_imgs):
            if f"pcd{filename}r.png" in cimg:
                return self.get_by_id(i)
        return None

    # -----------------------------------------------------
    # Internal: figure out which transform we apply
    #   transform_id in [0..11]
    # -----------------------------------------------------
    def __map_index(self, idx):
        """
        If augment=True, we did 'col_imgs * 12'. So total_len = original_len * 12.
        Then transform_id = idx // original_len, base_idx = idx % original_len.
        If augment=False, transform_id=0, base_idx=idx.
        """
        if not self.augment:
            return idx, 0
        base_idx = idx % self.original_len
        transform_id = idx // self.original_len  # 0..11
        return base_idx, transform_id

    # -----------------------------------------------------
    # These two apply the 12 transforms to the already
    # cropped images (300×300) and the corresponding grasps.
    # -----------------------------------------------------
    def __apply_augmentation_rgb(self, rgb_cropped, pos_grasps, transform_id, base_idx):
        """
        Transform the (300×300×3) color image. Return final shape (300×300×3).
        We do the same bounding‐box transform to 'pos_grasps' inside,
        so that if you need to check them, they match. 
        (In practice, the final bounding boxes are reloaded in __augment_grasp_list,
         but we keep everything consistent.)
        """
        # We’ll do a copy so as not to overwrite
        img = rgb_cropped
        if img.ndim == 2:
            # If somehow it's grayscale, add a channel dimension
            img = np.repeat(img[..., None], 3, axis=2)
        return self.__apply_transform_generic(
            img, pos_grasps, transform_id, base_idx, is_depth=False
        )

    def __apply_augmentation_depth(self, depth_cropped, pos_grasps, transform_id, base_idx):
        """
        Transform the (300×300) depth image. Return final shape (300×300).
        """
        img = depth_cropped[..., None]  # make it (H,W,1) so we can transform similarly
        out = self.__apply_transform_generic(
            img, pos_grasps, transform_id, base_idx, is_depth=True
        )
        # out => (300,300,1), return 2D
        return out[..., 0]

    def __augment_grasp_list(self, grasps, transform_id, base_idx):
        """
        We must do the exact same transform used for the image. 
        Easiest is to run `__apply_transform_generic` on a dummy array, because
        that function uses the same random margins, etc. Then we extract the
        new corners from the returned grasps.
        """
        # create dummy image 300×300×1 or ×3 (it doesn't matter as long as shape is consistent)
        dummy = np.zeros((self.__output_size[0], self.__output_size[1], 3), dtype=np.float32)
        self.__apply_transform_generic(dummy, grasps, transform_id, base_idx, is_depth=False)
        # grasps are now in their final position, so just return them
        return grasps

    # -------------------------------------------------------------
    # The heart of the 12 transformations:
    #   0) identity
    #   1) rotate 90
    #   2) rotate 180
    #   3) rotate 270
    #   4) random crop #1
    #   5) random crop #2
    #   6) zoom in
    #   7) zoom out
    #   8) vertical flip
    #   9) horizontal flip
    #   10) vertical flip + 90
    #   11) horizontal flip + 90
    # 
    # We apply them to a single 300×300 image (rgb or depth).
    # -------------------------------------------------------------
    def __apply_transform_generic(self, img, grasps, transform_id, base_idx, is_depth):
        """
        `img` is either (300,300,3) for RGB or (300,300,1) for depth.
        `grasps` is a list of Rectangle (4×2).
        We mutate `img` and `grasps` in place. Then return the resulting image.
        """
        H, W, C = img.shape

        # For “random” steps, we do a deterministic RNG seed = (base_idx + transform_id)
        # so each sample is stable across epochs.
        rng = np.random.RandomState(base_idx + transform_id)

        if transform_id == 0:
            # identity => do nothing
            return img

        elif transform_id == 1:
            # rotate 90° CCW
            # np.rot90 => shape (W,H,C), i.e. (300,300,C)
            out = np.rot90(img, k=1)
            self._rotate_grasps_90(grasps, center=(H/2, W/2))
            return out

        elif transform_id == 2:
            # rotate 180
            out = np.rot90(img, k=2)
            self._rotate_grasps_180(grasps, center=(H/2, W/2))
            return out

        elif transform_id == 3:
            # rotate 270° CCW
            out = np.rot90(img, k=3)
            self._rotate_grasps_270(grasps, center=(H/2, W/2))
            return out

        elif transform_id in [4, 5]:
            # random crop #1 or #2 => pick margins in 10..60, then resize to 300
            top_margin = rng.randint(10, 61)
            bot_margin = rng.randint(10, 61)
            left_margin = rng.randint(10, 61)
            right_margin = rng.randint(10, 61)

            row0, row1 = top_margin, H - bot_margin
            col0, col1 = left_margin, W - right_margin

            # ensure valid:
            if row1 <= row0 or col1 <= col0:
                # fallback to identity if margins are weird
                return img

            cropped = img[row0:row1, col0:col1]  # shape ~ (someH, someW, C)
            newH = row1 - row0
            newW = col1 - col0

            # scale back to (300,300)
            interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
            out = cv2.resize(cropped, (W, H), interpolation=interp)

            # transform grasps
            scale_h = H / float(newH)
            scale_w = W / float(newW)
            for g in grasps:
                pts = g.pts().astype(np.float32)
                pts[:, 0] -= row0
                pts[:, 1] -= col0
                pts[:, 0] *= scale_h
                pts[:, 1] *= scale_w
                pts[:, 0] = np.clip(pts[:, 0], 0, H-1)
                pts[:, 1] = np.clip(pts[:, 1], 0, W-1)
                g.set_pts(pts)
            return out

        elif transform_id == 6:
            # zoom in => e.g. crop a center region and upsize
            zoom_factor = 1.3  # you can tweak
            newH = int(H / zoom_factor)
            newW = int(W / zoom_factor)
            r0 = (H - newH)//2
            c0 = (W - newW)//2
            cropped = img[r0:r0+newH, c0:c0+newW]
            interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
            out = cv2.resize(cropped, (W, H), interpolation=interp)

            scale_h = H / float(newH)
            scale_w = W / float(newW)
            for g in grasps:
                pts = g.pts().astype(np.float32)
                pts[:, 0] -= r0
                pts[:, 1] -= c0
                pts[:, 0] *= scale_h
                pts[:, 1] *= scale_w
                pts[:, 0] = np.clip(pts[:, 0], 0, H-1)
                pts[:, 1] = np.clip(pts[:, 1], 0, W-1)
                g.set_pts(pts)
            return out

        elif transform_id == 7:
            # zoom out => shrink then place in center
            zoom_factor = 0.7
            small_h = int(H * zoom_factor)
            small_w = int(W * zoom_factor)
            interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
            shrinked = cv2.resize(img, (small_w, small_h), interpolation=interp)

            # put shrinked in center of black canvas
            out = np.zeros_like(img)
            r0 = (H - small_h)//2
            c0 = (W - small_w)//2
            if shrinked.ndim == 2 and out.ndim == 3 and out.shape[2] == 1:
                shrinked = np.expand_dims(shrinked, axis=-1)
            out[r0:r0+small_h, c0:c0+small_w] = shrinked

            # transform grasps
            for g in grasps:
                pts = g.pts().astype(np.float32)
                pts *= zoom_factor
                pts[:, 0] += r0
                pts[:, 1] += c0
                pts[:, 0] = np.clip(pts[:, 0], 0, H-1)
                pts[:, 1] = np.clip(pts[:, 1], 0, W-1)
                g.set_pts(pts)
            return out

        elif transform_id == 8:
            # vertical flip => flip over x-axis => row => H-1-row
            out = np.flip(img, axis=0)  # shape stays (300,300,C)
            for g in grasps:
                pts = g.pts()
                pts[:, 0] = (H - 1) - pts[:, 0]
                g.set_pts(pts)
            return out

        elif transform_id == 9:
            # horizontal flip => flip over y-axis => col => W-1-col
            out = np.flip(img, axis=1)
            for g in grasps:
                pts = g.pts()
                pts[:, 1] = (W - 1) - pts[:, 1]
                g.set_pts(pts)
            return out

        elif transform_id == 10:
            # vertical flip + rotate 90
            # do vertical flip
            flipped = np.flip(img, axis=0)
            for g in grasps:
                pts = g.pts()
                pts[:, 0] = (H - 1) - pts[:, 0]
                g.set_pts(pts)
            # then rotate 90
            out = np.rot90(flipped, k=1)
            self._rotate_grasps_90(grasps, center=(H/2, W/2))
            return out

        elif transform_id == 11:
            # horizontal flip + rotate 90
            # do horizontal flip
            flipped = np.flip(img, axis=1)
            for g in grasps:
                pts = g.pts()
                pts[:, 1] = (W - 1) - pts[:, 1]
                g.set_pts(pts)
            # then rotate 90
            out = np.rot90(flipped, k=1)
            self._rotate_grasps_90(grasps, center=(H/2, W/2))
            return out

        # Fallback (should never happen)
        return img

    # ---------------------------------
    # Helpers for rotating grasp boxes
    # Each grasp is 4 points [row,col]
    # ---------------------------------
    def _rotate_grasps_90(self, grasps, center):
        """
        90° CCW around `center=(cy, cx)`.
        (r, c) -> (r', c') = (- (c - cx), (r - cy)) + (cy, cx)
        """
        cy, cx = center
        for g in grasps:
            pts = g.pts().astype(np.float32)
            for i in range(len(pts)):
                r, c = pts[i]
                r_shift = r - cy
                c_shift = c - cx
                r_new = -c_shift
                c_new = r_shift
                pts[i, 0] = r_new + cy
                pts[i, 1] = c_new + cx
            g.set_pts(np.round(pts).astype(np.int32))

    def _rotate_grasps_180(self, grasps, center):
        """
        180° around center => (r,c)->(2*cy - r, 2*cx - c)
        """
        cy, cx = center
        for g in grasps:
            pts = g.pts().astype(np.float32)
            pts[:, 0] = 2 * cy - pts[:, 0]
            pts[:, 1] = 2 * cx - pts[:, 1]
            g.set_pts(np.round(pts).astype(np.int32))

    def _rotate_grasps_270(self, grasps, center):
        """
        270° CCW => or 90° CW => (r,c)->( (c - cx), -(r - cy) ) + (cy, cx)
        But simpler to do 3× 90° CCW or just do the direct approach:
        (r,c)->( c', r') with sign fix.
        """
        cy, cx = center
        for g in grasps:
            pts = g.pts().astype(np.float32)
            for i in range(len(pts)):
                r, c = pts[i]
                r_shift = r - cy
                c_shift = c - cx
                # 270° CCW => (r_shift, c_shift) -> (c_shift, -r_shift)
                r_new = c_shift
                c_new = -r_shift
                pts[i, 0] = r_new + cy
                pts[i, 1] = c_new + cx
            g.set_pts(np.round(pts).astype(np.int32))

