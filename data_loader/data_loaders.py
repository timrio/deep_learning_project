import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class EegDataset(Dataset):
    def __init__(self,
                 device,
                 features_eeg_path = './data/pre_processed_data/Multitaper_eeg_train.npy',
                 features_position_path = './data/pre_processed_data/Multitaper_position_train.npy',
                 target_path = './data/raw_data/y_train.csv'
                 ):
      
        # read features (ie multitaper)
        self.features_eeg = torch.tensor(np.abs(np.load(features_eeg_path)))
        self.features_position = torch.tensor(np.abs(np.load(features_position_path)))

        # read target
        if target_path:
          self.target = list(pd.read_csv(target_path, index_col = "index")['sleep_stage'])
          
        self.target_path = target_path

    def __len__(self):
        return self.features_eeg.shape[0]


    def __getitem__(self, idx):
        features_eeg = self.features_eeg[idx]
        features_position = self.features_position[idx].to(device, dtype=torch.float)
           
        if self.target_path is not None:
          target = torch.tensor(int(self.target[idx])).to(device)
          return (features_eeg, features_position), target
        return (features_eeg, features_position)