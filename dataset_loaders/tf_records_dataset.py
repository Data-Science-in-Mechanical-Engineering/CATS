import hydra
import omegaconf
import tensorflow as tf
# import tensorflow_datasets as tfds
from abc import ABC, abstractmethod


import functools
import pickle


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


def generator(data_source):
    for i in range(len(data_source)):
        yield data_source[i]


def create_tfrecords_dataset(data_source: TfSource, data_dir: str, name: str):

    output_signature = {"target": tf.TensorSpec(shape=(data_source.context_length + data_source.prediction_length, data_source.num_features), dtype=tf.float32)}
    
    ds_train = tf.data.Dataset.from_generator(functools.partial(generator, data_source.get_train_data_source()), 
                                        output_signature=output_signature)
    
    ds_val = tf.data.Dataset.from_generator(functools.partial(generator, data_source.get_val_data_source()), 
                                        output_signature=output_signature)
    ds_test = tf.data.Dataset.from_generator(functools.partial(generator, data_source.get_test_data_source()),
                                        output_signature=output_signature)    
    

    # Define the builder.
    single_number_builder = tfds.dataset_builders.TfDataBuilder(
        name=name,
        config=f"{data_source.context_length}_{data_source.prediction_length}_{data_source.stride}",
        version="1.0.0",
        data_dir=data_dir,
        split_datasets={
            "train": ds_train,
            "val": ds_val,
            "test": ds_test,
        },
        features=tfds.features.FeaturesDict({
            "target": tfds.features.Tensor(shape=(data_source.context_length + data_source.prediction_length, data_source.num_features), dtype=tf.float32),
        }),
        description="Test",
        release_notes={
            "1.0.0": "s",
        }
    )

    # Make the builder store the data as a TFDS dataset.
    single_number_builder.download_and_prepare()

    # save metadata
    metadata = {"normalization": data_source.get_normalization_params(), #
                "length_train": len(data_source.get_train_data_source()), 
                "length_val": len(data_source.get_val_data_source()), 
                "length_test": len(data_source.get_test_data_source())}
    with open(f"{data_dir}/{name}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
