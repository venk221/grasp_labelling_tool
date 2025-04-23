import graspnetAPI
import numpy as np
from skimage.transform import resize

from .grasp_dataset import GraspDataset
from .item import Rectangle, Item

IMAGE_SHAPE = (1280, 720)
CROP_START = (280, 0)
CROP_CENTER = (640, 360)
CROP_SIZE = (560, 0)
CROP_END = (IMAGE_SHAPE[0] - CROP_SIZE[0] // 2, IMAGE_SHAPE[1] - CROP_SIZE[1] // 2)

ANN_ID = 0


class Dataset(GraspDataset):
    def __init__(self, dataset_path, output_size=(300, 300), dep=True, rgb=True, start=0.0, end=1.0):
        super().__init__(output_size, dep, rgb)

        self.__output_size = output_size

        # Create a GraspNet object that can retrieve all the objects from the path provided
        self.__g = graspnetAPI.GraspNet(dataset_path, 'realsense')

    def __len__(self):
        return len(self.__g.sceneIds)

    def get_crop_params(self, grasps):
        means = np.zeros((len(grasps), 2))
        for idx, grasp in enumerate(grasps):
            means[idx] = np.mean(grasp.pts(), axis=0)

        center = np.flip(np.mean(means, axis=0, dtype=np.int))
        top = max(0, min(center[0] - self.__output_size[0] // 2, IMAGE_SHAPE[0] - self.__output_size[0]))
        left = max(0, min(center[1] - self.__output_size[1] // 2, IMAGE_SHAPE[1] - self.__output_size[1]))

        return center, top, left

    def get_grasp_file(self, idx):
        grasps = self.__g.loadGrasp(self.__g.sceneIds[idx], format='rect', camera='realsense', fric_coef_thresh=100.0, annId=ANN_ID)
        # Convert each grasp into a common format
        rectangles = []
        for grasp in grasps[:-1:2]:
            center = np.array(grasp.center_point)
            outer = np.array(grasp.open_point)
            direction = outer - center
            width = grasp.height
            length = np.linalg.norm(direction)
            direction /= length
            perpendicular = np.array([direction[1], -direction[0]])

            points = np.array([
                center + direction * length + perpendicular * width,
                center + direction * length - perpendicular * width,
                center - direction * length - perpendicular * width,
                center - direction * length + perpendicular * width
            ])

            if grasp.score > 0.5:
                rectangles.append(Rectangle(points, 1.0))
        return rectangles

    def get_raw_grasps(self, idx):
        grasps = self.get_grasp_file(idx)
        for grasp in grasps:
            grasp.set_pts(grasp.pts() + np.array([-CROP_SIZE[0] // 2, -CROP_SIZE[1]]))
            grasp.set_pts(grasp.pts() * self.__output_size / IMAGE_SHAPE[1])
        return grasps

    def crop(self, img, crop_params, crop_size):
        # Get center of all grasps
        center, left, top = crop_params

        return img[top:min(img.shape[0], top + crop_size[0]), left:min(img.shape[1], left + crop_size[1])]

    def __get_rgb(self, idx):
        mm_depth = self.__g.loadRGB(self.__g.sceneIds[idx], 'realsense', ANN_ID)
        mm_depth = mm_depth[CROP_START[1]:CROP_END[1], CROP_START[0]:CROP_END[0]]
        mm_depth = resize(mm_depth, self.__output_size, preserve_range=True).astype(mm_depth.dtype)
        mm_depth = mm_depth.astype(np.float32) / 255.0
        mm_depth -= mm_depth.mean()
        mm_depth = mm_depth.transpose((2, 0, 1))
        return mm_depth

    def __get_dep(self, idx):
        mm_depth = self.__g.loadDepth(self.__g.sceneIds[idx], 'realsense', ANN_ID)
        mm_depth = mm_depth[CROP_START[1]:CROP_END[1], CROP_START[0]:CROP_END[0]]
        mm_depth = resize(mm_depth, self.__output_size, preserve_range=True).astype(mm_depth.dtype)
        mm_depth = mm_depth.astype(np.float32) / 100.0
        mm_depth = np.resize(mm_depth, self.__output_size)
        return GraspDataset.normalize(mm_depth)

    def get_by_id(self, idx):
        ret = Item(self.__get_rgb(idx), self.__get_dep(idx), self.get_raw_grasps(idx))
        return ret

    def get_by_filename(self, filename):
        pass
