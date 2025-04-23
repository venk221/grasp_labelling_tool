from Datasets.grasp_dataset import GraspDataset


class Dataset(GraspDataset):
    def __init__(self, dataset_path, output_size=(300, 300), dep=True, rgb=False, start=0.0, end=1.0):
        super().__init__(output_size, dep, rgb)

        self.__start = start
        self.__end = end
