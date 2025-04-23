import os

from imageio import imread

from .item import *
from .grasp_dataset import GraspDataset


class Dataset(GraspDataset):
    def __init__(self, dataset_path, output_size=(300, 300), dep=True, rgb=False, start=0.0, end=1.0):
        super().__init__(output_size, dep, rgb)

        self.__dataset_path = dataset_path

        # Get a count of the total number of items available
        self.__files = []
        # Number of top-level directories
        outer_files = sorted(os.listdir(dataset_path))
        # For each of the above, get a count of the inner files
        for outer_file in outer_files:
            # The list of valid numbers within this outer directory
            inner_files = [inner_file[:inner_file.index('rgb')] for inner_file in sorted(os.listdir(os.path.join(dataset_path, outer_file))) if 'rgb' in inner_file]
            self.__files += [os.path.join(dataset_path, outer_file, inner_file) for inner_file in inner_files]
        self.__files = GraspDataset.split_list(self.__files, start, end)

    def __len__(self):
        return len(self.__files)

    def __load_grasps(self, idx):
        grasp_data = np.load(f'{self.__files[idx]}g.npy')
        
        # Construct grasp points for each grasp rectangle
        return [rect_shape_to_pts(grasp[:2], grasp[2], grasp[3], grasp[4], grasp[5]) for grasp in grasp_data]

    def __load_rgb(self, idx):
        # change imread to np.load
        return np.load(f'{self.__files[idx]}rgb.png.npy')

    def __load_dep(self, idx):
        # change imread to np.load
        return np.load(f'{self.__files[idx]}d.tiff.npy')

    def get_by_id(self, idx):
        return Item(self.__load_rgb(idx), self.__load_dep(idx), self.__load_grasps(idx))

    # Expects filename made of 5 digits
    def get_by_filename(self, filename):
        idx = self.__files.index(os.path.join(self.__dataset_path, self.filename[:2], self.filename[2:]))
        return self.get_by_id(idx)
