# https://authors.elsevier.com/sd/article/S221282711830307X

import os
import pandas as pd
from pathlib import Path
import numpy as np

from functools import partial

import tensorflow as tf
# import tensorflow_datasets as tfds

import pickle

from dataset_loaders.dataset_utils import normalize, denormalize, SplitMovingWindowDataSource
from dataset_loaders.tf_records_dataset import TfSource, create_tfrecords_dataset

from torch.utils.data import Dataset
from dataset_loaders.dataset_utils import DataSource


class ICDDataSource(DataSource):
    def __init__(self, context_length, prediction_length, dataset_base_path, name):
        super().__init__(name, context_length, prediction_length, 1, 8, Path(dataset_base_path) / "ICD")
        self.name = name
        csv_files = [file.name for file in self.dataset_base_path.iterdir() if file.suffix == '.csv' and file.is_file()]
       
        # months 1-8 are train, the rest val/test
        csv_files_train = [file for file in csv_files if file.startswith(tuple(f"0{i}" for i in range(9)))]
        csv_files_val = [file for file in csv_files if file.startswith(tuple(f"{i:02}" for i in range(9, 11)))]
        csv_files_test = [file for file in csv_files if file.startswith(tuple(f"{i:02}" for i in range(11, 13)))]

        # load data into RAM
        data_train = self.load_data(csv_files_train)
        data_val = self.load_data(csv_files_val)
        data_test = self.load_data(csv_files_test)

        # normalize
        data_train_total = np.concatenate(data_train, axis=0)
        self.mean = data_train_total.mean(axis=0)
        self.std = data_train_total.std(axis=0)

        data_train = [normalize(d, self.mean, self.std) for d in data_train]
        data_val = [normalize(d, self.mean, self.std) for d in data_val]
        data_test = [normalize(d, self.mean, self.std) for d in data_test]

        self.train_data_source = SplitMovingWindowDataSource(data_train, context_length, prediction_length)
        self.val_data_source = SplitMovingWindowDataSource(data_val, context_length, prediction_length)
        self.test_data_source = SplitMovingWindowDataSource(data_test, context_length, prediction_length)

        self.num_features = 8

    def load_data(self, csv_files):
        data = []
        for file in csv_files:
            df = pd.read_csv(self.dataset_base_path / file)
            data.append(np.array(df.iloc[:, 1:]))  # Skip the first column, which is the timestamp
        return data

    def get_train_data_source(self):
        return self.train_data_source

    def get_val_data_source(self):
        return self.val_data_source

    def get_test_data_source(self):
        return self.test_data_source
    
    def get_normalization_params(self):
        return {"mean": self.mean, "std": self.std}
    
    def denormalize(self, x):
        return denormalize(x, self.mean, self.std)
    

class ICD:
    def __init__(self, context_length, prediction_length, dataset_base_path, name):
        self.dataset_base_path = Path(dataset_base_path) / "ICD"

        # if the dataset is not already generated, generate it
        if not os.path.exists(self.dataset_base_path / f"icdtf/{context_length}_{prediction_length}"):
            print("Dataset not found as tfrecords. Generating tfrecords dataset")
            icd_data_source = ICDDataSource(context_length, prediction_length, self.dataset_base_path)
            create_tfrecords_dataset(data_source=icd_data_source,
                                     data_dir=self.dataset_base_path,
                                     name="icdtf")
            print("Created tfrecords file.")

        # load tfrecords dataset
        self.train_data_source = tfds.load(f"icdtf/{context_length}_{prediction_length}", split="train", data_dir=self.dataset_base_path)
        self.val_data_source = tfds.load(f"icdtf/{context_length}_{prediction_length}", split="val", data_dir=self.dataset_base_path)

        # load metadata
        metadata_file = self.dataset_base_path / f"icdtf/metadata.pkl"
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        self.mean = metadata["normalization"]["mean"]
        self.std = metadata["normalization"]["std"]
        self.length_train = metadata["length_train"]
        self.length_val = metadata["length_val"]
        self.length_test = metadata["length_test"]

        self.denormalize = partial(denormalize, mean=self.mean, std=self.std)
        self.num_features = 8


if __name__ == "__main__":
    icd = ICD(512, 128, '/data/datasets/industrial_component_degradation', 'ICD')
    print(len(icd.train_data_source))
    print(icd.train_data_source[20000])
