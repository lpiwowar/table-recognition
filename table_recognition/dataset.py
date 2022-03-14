import os

import torch
from torch_geometric.data import Dataset
from torch.utils.data import Dataset


class TableDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.data_files = os.listdir(self.config.train_data_dir)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, item):
        data_file = self.data_files[item]
        return torch.load(os.path.join(self.config.train_data_dir, data_file))


class TableDataset2(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, config=None):
        print(root)
        super().__init__(root, transform, pre_transform, pre_filter)
        self.config = config

    @property
    def raw_file_names(self):
        return os.listdir(self.config.train_data_dir)

    @property
    def processed_file_names(self):
        return os.listdir(self.config.train_data_dir)

    def download(self):
        raise Exception("ERROR: Download for TableDataset not implemented!")

    def process(self):
        pass
        #idx = 0
        #for raw_path in self.raw_paths:
            # Read data from `raw_path`.
       #     data = Data(...)

       #     if self.pre_filter is not None and not self.pre_filter(data):
       #         continue

       #     if self.pre_transform is not None:
       #         data = self.pre_transform(data)

       #     torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
       #     idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data