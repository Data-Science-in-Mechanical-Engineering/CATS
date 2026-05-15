import grain.python as grain
import pandas as pd
from pathlib import Path
import numpy as np

from functools import partial

import tensorflow as tf

from dataset_loaders.dataset_utils import normalize, denormalize

class PeriodicDataSource(grain.RandomAccessDataSource):
    def __init__(self, data, context_length, prediction_length):
        self.__data = data
        self.__context_length = context_length
        self.__prediction_length = prediction_length
    
    def __getitem__(self, idx):
        return  {"input": self.__data[idx, :self.__context_length],
                "target": self.__data[idx, :]}

    def __len__(self):
        return len(self.__data) - self.__context_length - self.__prediction_length + 1

class Periodic:
    def __init__(self, context_length, prediction_length, name, dataset_base_path):
        num_data_points = 100000
        frequencies = np.random.uniform(0.0, 0.1, (num_data_points, 10))
        delta = np.random.uniform(0.0, 2 * np.pi, (num_data_points, 10))


        data = np.zeros((num_data_points, context_length + prediction_length, 1))
        for i, f in enumerate(frequencies):
            for j, f_i in enumerate(f):
                data[i, :, 0] += np.sin(2 * np.pi * np.arange(context_length + prediction_length) * f_i + delta[i, j])
        
        data_train = data[:int(0.6*len(data))]
        data_val = data[int(0.6*len(data)):int(0.8*len(data))]
        data_test = data[int(0.8*len(data)):]

        self.__mean = data_train.mean()
        self.__std = data_train.std()

        data_train = normalize(data_train, self.__mean, self.__std)
        data_val = normalize(data_val, self.__mean, self.__std)
        data_test = normalize(data_test, self.__mean, self.__std)

        self.train_data_source = tf.data.Dataset.from_tensor_slices({"target": data_train})
        self.val_data_source = tf.data.Dataset.from_tensor_slices({"target": data_val})
        self.test_data_source = tf.data.Dataset.from_tensor_slices({"target": data_test})

        self.denormalize = partial(denormalize, mean=self.__mean, std=self.__std)
        self.num_features = 1

        self.length_train = len(data_train)
        self.length_val = len(data_val)
        print(f"Initialized dataset {name}")

    