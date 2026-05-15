from abc import ABC, abstractmethod
import functools
import logging
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


from dataset_loaders.rand_augment import distort_image_with_randaugment

def normalize(data, mean, std):
    return (data - mean) / std

def denormalize(data, mean, std):
    return data * std + mean


class DataSource(ABC):

    def __init__(self, name, context_length, prediction_length, stride, num_features, dataset_base_path):
        self.name = name
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

    @property
    def length_train(self):
        return len(self.get_train_data_source())
    
    @property
    def length_val(self):
        return len(self.get_val_data_source()) 
    
    @property
    def length_test(self):
        return len(self.get_test_data_source())
    
    @abstractmethod
    def denormalize(self, x):
        pass

    @abstractmethod
    def get_normalization_params(self):
        pass


class SplitMovingWindowDataSource(Dataset):
    """
    Data source for one split (e.g., train, val, test) of the ICD dataset.
    """
    def __init__(self, data, context_length, prediction_length, stride=1):
        self.__data = data
        lengths = [len(d) for d in data]
        num_datapoints = [(l - context_length - prediction_length) // stride + 1 for l in lengths]
        self.__indexes = np.cumsum(num_datapoints)
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
    
    def __getitem__(self, idx):
        data_idx = np.searchsorted(self.__indexes, idx, side="right")
        if data_idx == 0:
            index_in_data = idx
        else:
            index_in_data = idx - self.__indexes[data_idx-1] 
        index_in_data *= self.stride
        return  {"target": self.__data[data_idx][index_in_data:index_in_data + self.context_length + self.prediction_length, :],} 
                 # "input": self.__data[data_idx][index_in_data:index_in_data + self.context_length, :]}

    def __len__(self):
        return self.__indexes[-1]
    

def init_torch_dataloader(data_source, batch_size, num_workers, do_distributed_training, num_nodes):
    logger = logging.getLogger("execution_guard")
    if do_distributed_training:
        logger.info(f"Using distributed training with {num_nodes} nodes.")
        dist_sampler = DistributedSampler(data_source, shuffle=True)
    else:
        dist_sampler = None

    assert batch_size % num_nodes == 0, f"Batch size {batch_size} must be divisible by number of nodes {num_nodes} for distributed training."
    
    if dist_sampler is not None:
        return DataLoader(data_source, 
                        batch_size=batch_size // num_nodes, 
                        num_workers=num_workers, 
                        prefetch_factor=None,
                        persistent_workers=True,
                        sampler=dist_sampler,
                        ), dist_sampler
    else:
        return DataLoader(data_source, 
                        batch_size=batch_size, 
                        num_workers=num_workers, 
                        prefetch_factor=None,
                        persistent_workers=True,
                        shuffle=True), None

def generator(data_source):
    for i in range(len(data_source)):
        yield data_source[i]


def init_tf_dataloader_timeseries(data_source, batch_size, num_epochs, seed, num_features, prediction_length, context_length, num_workers=1):
    output_signature = {"target": tf.TensorSpec(shape=(context_length + prediction_length, num_features), dtype=tf.float32)}
    dataset = tf.data.Dataset.from_generator(
        generator=functools.partial(generator, data_source),
        output_signature=output_signature
        )
    data = dataset.shuffle(10000, seed=seed)
    data = data.batch(batch_size, num_parallel_calls=num_workers, drop_remainder=True)
    data = data.repeat(num_epochs + 1000)
    data = data.prefetch(2)
    data = data.as_numpy_iterator()
    return data


def random_cropping(image):
    size_mult = tf.random.uniform([], minval=0.08, maxval=1.0)
    aspect_ratio = tf.random.uniform([], minval=3 / 4, maxval=4 / 3)
    size_mult = tf.cond(aspect_ratio > 1, lambda: tf.stack([size_mult, size_mult / aspect_ratio]), lambda: tf.stack([size_mult * aspect_ratio, size_mult]))
    image_shape = tf.cast(tf.shape(image), dtype=tf.float32)
    new_shape = [tf.clip_by_value(size_mult[0] * image_shape[0], clip_value_min=1, clip_value_max=image_shape[0]), 
                 tf.clip_by_value(size_mult[1] * image_shape[1], clip_value_min=1, clip_value_max=image_shape[1]), 
                 3]

    return tf.image.random_crop(image, size=new_shape)

def init_tf_dataloader_image(data_source, batch_size, num_epochs, seed, rand_augment_functions, resolution, num_parallel_calls=tf.data.AUTOTUNE):
    data = data_source.shuffle(10000, seed=seed, reshuffle_each_iteration=True)
    data = data.map(lambda x: {"input": x["image"], "target": x["label"]}, num_parallel_calls=num_parallel_calls)
    # data = data.map(lambda x: {"input": tf.image.resize(x["input"], (resolution, resolution)), "target": x["target"]}, num_parallel_calls=num_parallel_calls)
    if rand_augment_functions is not None:
        data = data.map(lambda x: {"input": tf.image.random_flip_left_right(random_cropping(x["input"])), "target": x["target"]}, 
                    num_parallel_calls=num_parallel_calls)
        data = data.map(lambda x: {"input": distort_image_with_randaugment(image=tf.cast(x["input"], dtype=tf.float32), 
                                                                            num_layers=2, functions=rand_augment_functions), "target": x["target"]}, num_parallel_calls=num_parallel_calls)
    data = data.map(lambda x: {"input": (tf.image.resize(x["input"], (resolution, resolution)) / 255 - 0.5) * 2, "target": x["target"]}, num_parallel_calls=num_parallel_calls)
    data = data.batch(batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls)
    data = data.repeat(num_epochs + 1000)
    data = data.prefetch(2)
    data = data.as_numpy_iterator()
    return data


