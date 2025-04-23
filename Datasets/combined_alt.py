import os

from torch.utils.data import Dataset as TorchDataset

from Datasets.cornell import Dataset as CornellDataset
from Datasets.jacquard import Dataset as JacquardDataset


class Dataset(TorchDataset):
    def __init__(self, data_dir, dep=True, rgb=False,
                 start=0.0, end=1.0, rebalance_datasets=True):  # changed rebalance_datasets to False
        super().__init__()

        # Create sub-datasets
        self.__cornell_dataset = CornellDataset(os.path.join(data_dir, 'Cornell'), dep=dep, rgb=rgb, # changed from Cornell
                                                start=start, end=end)
        self.__jacquard_dataset = JacquardDataset(os.path.join(data_dir, 'JacquardDataset'), dep=dep, rgb=rgb, # changed from Jacquard
                                                  start=start, end=end)

        if rebalance_datasets:
            lc = len(self.__cornell_dataset)
            lj = len(self.__jacquard_dataset)
            original_size = lj / (end - start)
            required_scale = lc / original_size

            self.__jacquard_dataset = JacquardDataset(os.path.join(data_dir, 'JacquardDataset'), dep=dep, rgb=rgb, # changed from Jacquard
                                                      start=end - required_scale, end=end)

        print(f'Created combined dataset with {len(self.__cornell_dataset)} cornell and {len(self.__jacquard_dataset)} jacquard points')

    def __len__(self):
        return len(self.__cornell_dataset) + len(self.__jacquard_dataset)

    def __getitem__(self, item):
        if item >= len(self.__cornell_dataset):
            x, y, idx, rot, zoom = self.__jacquard_dataset[item - len(self.__cornell_dataset)]
            # Offset the idx returned
            idx += len(self.__cornell_dataset)
            return x, y, idx, rot, zoom
        else:
            return self.__cornell_dataset[item]

    def get_grasps(self, item):
        if item >= len(self.__cornell_dataset):
            return self.__jacquard_dataset.get_grasps(item - len(self.__cornell_dataset))
        else:
            return self.__cornell_dataset.get_grasps(item)
