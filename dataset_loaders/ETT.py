import grain.python as grain
import pandas as pd
from pathlib import Path
import numpy as np

from functools import partial
from torch.utils.data import Dataset

from dataset_loaders.dataset_utils import normalize, denormalize
from dataset_loaders.dataset_utils import DataSource

class ETTDataScource(Dataset):
    def __init__(self, data, context_length, prediction_length):
        self.__data = data
        self.__context_length = context_length
        self.__prediction_length = prediction_length
    
    def __getitem__(self, idx):
        data = self.__data[idx:idx+self.__context_length+self.__prediction_length]
        data = data.reshape(-1, 1)  # Ensure data is 2D

        return  {"target": data}

    def __len__(self):
        return len(self.__data) - self.__context_length - self.__prediction_length + 1

class ETT(DataSource):
    def __init__(self, variant, context_length, prediction_length, dataset_base_path, name):
        print(Path(dataset_base_path) / f"ETT/ETT{variant}.csv")
        df = pd.read_csv(Path(dataset_base_path) / f"ETT/ETT{variant}.csv")
        self.__data = np.array(df["OT"])
        self.__data_train = self.__data[:int(0.6*len(self.__data))]
        self.__data_val = self.__data[int(0.6*len(self.__data)):int(0.8*len(self.__data))]
        self.__data_test = self.__data[int(0.8*len(self.__data)):]
        
        self.__mean = self.__data_train.mean()
        self.__std = self.__data_train.std()

        self.__data_train = normalize(self.__data_train, self.__mean, self.__std)
        self.__data_val = normalize(self.__data_val, self.__mean, self.__std)
        self.__data_test = normalize(self.__data_test, self.__mean, self.__std)

        self.train_data_source = ETTDataScource(self.__data_train, context_length, prediction_length)
        self.val_data_source = ETTDataScource(self.__data_val, context_length, prediction_length)#
        self.test_data_source = ETTDataScource(self.__data_test, context_length, prediction_length)

        print(f"Initialized dataset {name}")

        self.num_features = 1

    def get_train_data_source(self):
        return self.train_data_source

    def get_val_data_source(self):
        return self.val_data_source

    def get_test_data_source(self):
        return self.test_data_source
    
    def get_normalization_params(self):
        return {"mean": self.__mean, "std": self.__std}
    
    def denormalize(self, x):
        return denormalize(x, self.__mean, self.__std)
    