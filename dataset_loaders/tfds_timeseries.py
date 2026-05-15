from pathlib import Path
import datasets
import hydra
import numpy as np
import omegaconf
import tensorflow as tf
import tensorflow_datasets as tfds
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod


import functools
import pickle

from tqdm import tqdm
from collections import Counter

from dataset_loaders.utsdataset import UTSDataSource

tf.config.set_visible_devices([], 'GPU')


class TfSource(ABC):

    def __init__(self, context_length, prediction_length, stride, num_features, dataset_base_path):
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride

        self.num_features = num_features
        self.dataset_base_path = dataset_base_path


    @abstractmethod
    def get_train_data_source(self):
        pass

    @abstractmethod
    def get_val_data_source(self):
        pass

    @abstractmethod
    def get_test_data_source(self):
        pass

    @abstractmethod
    def get_normalization_params(self):
        pass

class SplitUTSDatasetSource:
    def __init__(self, seq_length, stride, flag, dataset_base_path, split=0.9, scale=True, subset_name='UTSD-1G'):
        self.seq_length = seq_length
        self.stride = stride
        self.split = split
        self.scale = scale
        self.subset_name = subset_name
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.dataset_base_path = dataset_base_path
        dataset = datasets.load_dataset("thuml/UTSD", self.subset_name, split='train', cache_dir=self.dataset_base_path)

        self.data_list = []
        self.data_lengths = []
        print('Indexing dataset...')
        for i, item in tqdm(enumerate(dataset)):
            # if i > 10000:
            #     break
            self.scaler = StandardScaler()
            data = item['target']
            data = np.array(data).reshape(-1, 1)
            num_train = int(len(data) * self.split)
            border1s = [0, num_train - self.seq_length]
            border2s = [num_train, len(data)]

            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = self.scaler.transform(data)

            data = data[border1:border2]
            n_window = (len(data) - self.seq_length) // self.stride + 1
            if n_window < 1:
                continue
            self.data_lengths.append(len(data))
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return {"target": tf.convert_to_tensor(self.data_list[idx], dtype=tf.float32)}



class UTSDatasetSource(TfSource):
    def __init__(self, context_length, prediction_length, stride, num_features, dataset_base_path, split=0.9):
        super().__init__(context_length, prediction_length, stride, num_features, dataset_base_path)

        self.train_data_source = SplitUTSDatasetSource(seq_length=context_length + prediction_length, 
                                                       stride=stride, flag='train', 
                                                       dataset_base_path=dataset_base_path, 
                                                       split=split, 
                                                       scale=True)
        self.val_data_source = SplitUTSDatasetSource(seq_length=context_length + prediction_length, 
                                                     stride=stride, flag='val', 
                                                     dataset_base_path=dataset_base_path, 
                                                     split=split, 
                                                     scale=True)


    def get_train_data_source(self):
        return self.train_data_source

    def get_val_data_source(self):
        return self.val_data_source

    def get_test_data_source(self):
        return self.val_data_source  # For now, we use the validation data as test data

    def get_normalization_params(self):
        return {"mean": 0, "std": 1}


def generator(data_source):
    for i in range(len(data_source)):
        yield data_source[i]


def create_tfrecords_dataset(data_source: TfSource, data_dir: str, name: str):

    output_signature = {"target": tf.TensorSpec(shape=(None, data_source.num_features), dtype=tf.float32)}

    # Calculate distribution of number of windows in training data
    windows_list = [
        length_ // (data_source.context_length + data_source.prediction_length)
        for length_ in data_source.get_train_data_source().data_lengths
    ]

    max_num_windows = max(windows_list)

    initial_distribution = []
    for i in range(1, max_num_windows + 1):
        initial_distribution.append(windows_list.count(i))

    initial_distribution = np.array(initial_distribution, dtype=np.float32)
    initial_distribution = (initial_distribution / np.sum(initial_distribution)).tolist()  # Normalize to sum to

    
    ds_train = tf.data.Dataset.from_generator(functools.partial(generator, data_source.get_train_data_source()), 
                                        output_signature=output_signature)
    
    ds_val = tf.data.Dataset.from_generator(functools.partial(generator, data_source.get_val_data_source()), 
                                        output_signature=output_signature)
    ds_test = tf.data.Dataset.from_generator(functools.partial(generator, data_source.get_test_data_source()),
                                        output_signature=output_signature)    
    

    # Define the builder.
    # single_number_builder = tfds.dataset_builders.TfDataBuilder(
    #     name=name,
    #     config=f"{data_source.context_length}_{data_source.prediction_length}_{data_source.stride}",
    #     version="1.0.0",
    #     data_dir=data_dir,
    #     split_datasets={
    #         "train": ds_train,
    #         "val": ds_val,
    #         "test": ds_test,
    #     },
    #     features=tfds.features.FeaturesDict({
    #         "target": tfds.features.Sequence(feature=tfds.features.Tensor(shape=(data_source.num_features,), dtype=tf.float32))
    #     }),
    #     description="Test",
    #     release_notes={
    #         "1.0.0": "s",
    #     }
    # )

    # Make the builder store the data as a TFDS dataset.
    # single_number_builder.download_and_prepare()

    # save metadata
    metadata = {"normalization": data_source.get_normalization_params(), #
                "length_train": len(data_source.get_train_data_source()), 
                "length_val": len(data_source.get_val_data_source()), 
                "length_test": len(data_source.get_test_data_source()),
                "initial_distribution": initial_distribution,}
    with open(f"{data_dir}/{name}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


def random_window(ts, window_length):
    length = tf.shape(ts)[0]
    start = tf.random.uniform([], minval=0, maxval=length - window_length + 1, dtype=tf.int32)
    return ts[start : start + window_length]


def init_tf_dataloader_moving_window_timeseries(data_source, batch_size, num_epochs, seed, context_length, prediction_length, initial_distribution):
    max_num_windows = len(initial_distribution)
    target_dist = np.arange(1, max_num_windows + 1, dtype=np.float32)
    target_dist = target_dist / np.sum(target_dist)  # Normalize to sum to

    initial_target_ratio = [t/i for i, t in zip(initial_distribution, target_dist)]
    data = data_source.repeat(max(int(max(initial_target_ratio)), 1))
    data = data.rejection_resample(class_func=lambda x: tf.minimum(tf.shape(x["target"])[0] // (context_length + prediction_length), max_num_windows)-1, 
                                            target_dist=target_dist,
                                            initial_dist=initial_distribution)
    
    data = data.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
    # data = data.map(lambda x: {"target": random_window(x["target"], context_length+prediction_length)}, num_parallel_calls=tf.data.AUTOTUNE)
    # data = data.map(lambda x: {"input": x["target"][0:context_length, :], "target": x["target"]}, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
    # data = data.repeat(num_epochs + 1000)
    data = data.prefetch(1)
    data = data.as_numpy_iterator()
    return data


def init_tf_dataset_timeseries(data_source, batch_size, num_epochs, seed, context_length,):
    data = data_source.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
    # data = data.map(lambda x: {"target": random_window(x["target"], context_length+prediction_length)}, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.map(lambda x: {"input": x["target"][0:context_length, :], "target": x["target"]}, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
    data = data.repeat(num_epochs + 1000)
    data = data.prefetch(1)
    data = data.as_numpy_iterator()
    return data



if __name__ == "__main__":
    data_source = UTSDataSource(dataset_base_path='/data/datasets/', subset_name=r'UTSD-1G', flag='train', split=0.9, context_length=512, prediction_length=96)
    output_signature = {"target": tf.TensorSpec(shape=(None, 1), dtype=tf.float32)}
    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator, data_source),
        output_signature=output_signature)
    
    dataset = init_tf_dataset_timeseries(dataset,
                                         batch_size=1024,
                                         num_epochs=10,
                                         seed=42,
                                         context_length=512)
    
    i = 0
    shapes = []
    for batch in tqdm(dataset, desc="Dataset1"):
        i += 1

    print(i)

    exit()

    d = UTSDatasetSource(context_length=512, prediction_length=96, stride=1, num_features=1, dataset_base_path='/data/datasets/', split=0.9)
    
    max_num_windows = 10
    # d = UTSDatasetSource(context_length=512, prediction_length=96, stride=1, num_features=1, dataset_base_path='/data/datasets/', split=0.9)

    # create_tfrecords_dataset(data_source=d,
    #                          data_dir=Path(d.dataset_base_path) / "UTSD",
    #                          name="utsd")
    
    train_data_source = tfds.load(f"utsd/{512}_{96}_{1}", split="train", data_dir='/data/datasets/UTSD')
    val_data_source = tfds.load(f"utsd/{512}_{96}_{1}", split="val", data_dir='/data/datasets/UTSD')

    metadata_file = Path("/data/datasets/UTSD") / "utsd" / "metadata.pkl"
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    print("Loaded metadata:", metadata)

    max_num_windows = min(max_num_windows, len(metadata["initial_distribution"]))

    initial_distribution = metadata["initial_distribution"][:max_num_windows]
    initial_distribution[-1] = 1.0 - sum(initial_distribution[:-1])  # Ensure the last value is set to make the sum equal to 1
    print(initial_distribution)

    train_dataset = init_tf_dataloader_moving_window_timeseries(train_data_source, 
                                                                batch_size=1, 
                                                                num_epochs=10, 
                                                                seed=42, 
                                                                context_length=512, 
                                                                prediction_length=96,
                                                                initial_distribution=initial_distribution,)
    
    initial_distribution = [1.0]
    train_dataset2 = init_tf_dataloader_moving_window_timeseries(train_data_source, 
                                                                batch_size=1, 
                                                                num_epochs=10, 
                                                                seed=42, 
                                                                context_length=512, 
                                                                prediction_length=96,
                                                                initial_distribution=initial_distribution,)

    i = 0
    shapes = []
    for batch in tqdm(train_dataset, desc="Dataset1"):
        # print(batch)
        shapes.append(batch[1]["target"].shape)
        i += 1

    shapes2 = []
    for batch in tqdm(train_dataset2, desc="Dataset2"):
        shapes2.append(batch[1]["target"].shape)
        i += 1
    
    print(i)

    import matplotlib.pyplot as plt

    # Extract the first dimension from each shape (i.e., sequence length)
    seq_lengths = [s[1] for s in shapes]
    seq_lengths2 = [s[1] for s in shapes2]

    plt.figure(figsize=(8, 6))
    plt.hist(seq_lengths, bins=1000, edgecolor='blue', density=True)
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.title("Histogram of Sequence Lengths")

    plt.figure(figsize=(8, 6))
    plt.hist(seq_lengths2, bins=1000, edgecolor='blue', density=True)
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.title("Histogram of Sequence Lengths")
    plt.show()

