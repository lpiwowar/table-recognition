import os

import torch
from torch.utils.data import Dataset


class TableDataset(Dataset):
    def __init__(self, config, datatype):
        assert datatype == "train" or datatype == "valid" or datatype == "test"

        self.config = config
        self.datatype = datatype

        with open(self.config.valid_list, "r") as f:
            valid_file_names = f.readlines()

        self.valid_list = [os.path.join(self.config.data_dir, valid_file_name)
                           for valid_file_name in valid_file_names]

        with open(self.config.test_list, "r") as f:
            test_file_names = f.readlines()

        self.test_list = [os.path.join(self.config.test_dir, test_file_name)
                          for test_file_name in test_file_names]

        with open(self.config.train_list, "r") as f:
            train_file_names = f.readlines()

        self.train_list = [os.path.join(self.config.test_dir, train_file_name)
                           for train_file_name in train_file_names]

        if self.datatype == "train":
            self.data_files = self.train_list
        elif self.datatype == "valid":
            self.data_files = self.valid_list
        elif self.datatype == "test":
            self.data_files = self.test_list
        else:
            self.data_files = None

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, item):
        data_file = self.data_files[item]

        return torch.load(os.path.join(self.config.data_dir, data_file))

    @property
    def raw_file_names(self):
        return os.listdir(self.config.train_data_dir)
