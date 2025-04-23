import os
import random
from torch.utils.data import Dataset as TorchDataset
from Datasets.cornell import CornellDataset #as CornellDataset
from Datasets.jacquard import JacquardDataset # as JacquardDataset

class Dataset(TorchDataset):
    def __init__(self, data_dir, output_size=(300, 300),  dep=True, rgb=False,
                 start=0.0, end=1.0, rebalance_datasets=True, seed=65, augment=False, max_samples=None):  # changed rebalance_datasets to False
        super().__init__()

        random.seed(seed)
        self.augment = augment
        self.max_samples = max_samples
        self.output_size = output_size
        # Create sub-datasets
        print("Cornell")
        self.__cornell_dataset = CornellDataset(os.path.join(data_dir, 'CornellGraspDataset'), output_size=self.output_size, dep=dep, rgb=rgb, # changed from Cornell
                                                start=start, end=end, augment=self.augment, max_samples=self.max_samples)
        print("Jacquard")
        self.__jacquard_dataset = JacquardDataset(os.path.join(data_dir, 'JacquardDataset'), output_size=self.output_size, dep=dep, rgb=rgb, # changed from Jacquard
                                                  start=start, end=end, augment=self.augment, max_samples=None)
        self.__rebalance_datasets = rebalance_datasets
        if rebalance_datasets:
            lc = len(self.__cornell_dataset)
            lj = len(self.__jacquard_dataset)
            
            population = list(range(0, lj))
            self.__random_samples_jacquard_dataset = random.sample(population, lc)
            
            # original_size = lj / (end - start)
            # required_scale = lc / original_size
            # self.__jacquard_dataset = JacquardDataset(os.path.join(data_dir, 'JacquardDataset'), dep=dep, rgb=rgb, # changed from Jacquard
            #                                           start=end - required_scale, end=end)

        print(f'Created combined dataset with {len(self.__cornell_dataset)} cornell and {len(self.__jacquard_dataset)} jacquard points')

    def __len__(self):
        return len(self.__cornell_dataset) + len(self.__random_samples_jacquard_dataset)

    def __getitem__(self, item):
        if item >= len(self.__cornell_dataset):
            if self.__rebalance_datasets:
                x, y, idx, rot, zoom = self.__jacquard_dataset[
                    self.__random_samples_jacquard_dataset[item - len(self.__cornell_dataset)]]
            else:
                x, y, idx, rot, zoom = self.__jacquard_dataset[item - len(self.__cornell_dataset)]
            # Offset the idx returned
            # idx += len(self.__cornell_dataset)
            idx = item
            return x, y, idx, rot, zoom
        else:
            return self.__cornell_dataset[item]

    def get_grasps(self, item):
        try:
            if item >= len(self.__cornell_dataset):
                # return self.__jacquard_dataset.get_grasps(item - len(self.__cornell_dataset))
                if self.__rebalance_datasets:
                    return self.__jacquard_dataset.get_grasps(self.__random_samples_jacquard_dataset[item - len(self.__cornell_dataset)])
                else:
                    return self.__jacquard_dataset.get_grasps(item - len(self.__cornell_dataset))
            else:
                return self.__cornell_dataset.get_grasps(item)
        except:
            print(f"item {item} ")
    
    def get_by_id(self, idx):
        """Alias for get_grasps to maintain compatibility"""
        return self.get_grasps(idx)
    

