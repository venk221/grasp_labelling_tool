import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

# ''GGCNN_Cornell','GGCNN_Jacquard','GRCNN_Cornell','GRCNN_Jacquard','GPNN_Cornell','GPNN_Jacquard''
class ExpertOutDataset(Dataset):
    def __init__(self, expert_out_dirs, start=0.0, end=1.0):
        self.__expert_out_dirs = expert_out_dirs
        self.__start = start
        self.__end = end

        # Read the expert names and use them for blacklisting
        self.__expert_names = open(os.path.join(self.__expert_out_dirs, 'experts.txt')).read().split(',')
        # self.__expert_names = None
        print(f'Loading dataset containing output of {self.__expert_names}')
        # Setup files for loading
        self.__data = []
        # We expect the output directories to be 0 -> len(dataset), so we can represent that with a range
        # We have a single non-directory file (experts.txt) so subtract one to get the actual count
        # This makes sure the data is sorted as well
        for idx in range(len(os.listdir(self.__expert_out_dirs)) - 1):
            # if idx == 100:
            #     break
            idx = str(idx)
            # Store them here
            self.__data.append({'experts': [], 'x': None, 'y': None, 'grasps': None})

            # First, load the input data
            self.__data[-1]['x'] = os.path.join(self.__expert_out_dirs, idx, 'data.npy')

            # Second, load the ground truth data
            self.__data[-1]['y'] = [os.path.join(self.__expert_out_dirs, idx, 'gt_pos.npy'),
                                    os.path.join(self.__expert_out_dirs, idx, 'gt_sin.npy'),
                                    os.path.join(self.__expert_out_dirs, idx, 'gt_cos.npy'),
                                    os.path.join(self.__expert_out_dirs, idx, 'gt_wid.npy')]

            # Third, load the grasp data
            self.__data[-1]['grasps'] = os.path.join(self.__expert_out_dirs, idx, 'gt_gsp.pickle')

            # Finally, load the expert data
            # Again, we expect {expert_id}[q|a|w].npy, so make a range going from 0 -> len(*q.npy files)
            
            # the main one
            for expert_id in range(len(glob.glob(os.path.join(self.__expert_out_dirs, idx, '*q.npy')))):
                self.__data[-1]['experts'].append(
                    {'q': os.path.join(self.__expert_out_dirs, idx, f'{expert_id}q.npy'),
                     'a': os.path.join(self.__expert_out_dirs, idx, f'{expert_id}a.npy'),
                     'w': os.path.join(self.__expert_out_dirs, idx, f'{expert_id}w.npy')})
            
        # Trim according to the start and end splits
        self.__data = self.__data[int(len(self.__data) * self.__start): int(len(self.__data) * self.__end)]
        print(f'Created expert dataset of size {len(self.__data)}')

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        data = self.__data[idx]
        # Combine expert and input data into one large tensor
        # This tensor has dimensions (x channels + expert_count * 3)
        img = np.load(data['x'])
        image_size = img.shape[1:]
        
        expert_tensor = torch.zeros((img.shape[0] + len(data['experts']) * 3, *image_size), dtype=torch.float32)
        for expert_idx in range(len(data['experts'])):
            # if expert_idx in self.__blacklisted_experts:
            #     continue
            expert_tensor[expert_idx * 3 + 0] = torch.from_numpy(np.load(data['experts'][expert_idx]['q']))
            expert_tensor[expert_idx * 3 + 1] = torch.from_numpy(np.load(data['experts'][expert_idx]['a']))
            expert_tensor[expert_idx * 3 + 2] = torch.from_numpy(np.load(data['experts'][expert_idx]['w']))
    
        expert_tensor[-img.shape[0]:] = torch.from_numpy(img)
        
        y = [np.load(f) for f in data['y']]

        # Return input and ground truth data
        return expert_tensor, y, idx

    def get_expert_names(self):
        return self.__expert_names
        
    def get_grasps(self, idx):
        return pickle.load(open(self.__data[idx]['grasps'], 'rb'))
