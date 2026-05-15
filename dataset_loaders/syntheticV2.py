
from functools import partial
import os
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import tensorflow_datasets as tfds

from dataset_loaders.dataset_utils import denormalize, init_tf_dataloader_timeseries

def sample_polynomial(x, degree=6):
    """Sample a polynomial function of a given degree."""
    coeffs = np.random.uniform(-1, 1, degree + 1)

    max_x = np.max(np.abs(x))
    # Normalize x to the range [-1, 1] for better polynomial behavior
    for i in range(len(coeffs)):
        coeffs[i] /= (max_x ** (len(coeffs)-i-1))

    return np.polyval(coeffs, x)

def sample_sine(x):
    return np.sin(x)

def sample_tan(x):
    """Sample a tangent function."""
    return np.tan(x)

def sample_exp(x):
    """Sample an exponential function."""
    x /= np.max(np.abs(x))
    base = np.random.uniform(-2.0, 1.0)  # Random base for the exponential
    return np.exp(base * x)

def sample_abs_log(x):
    """Sample a logarithmic function."""
    return np.clip(np.log(np.abs(x) + 1e-5), a_min=-10, a_max=np.inf)  # Adding a small constant to avoid log(0)

def sample_relu(x):
    """Sample a ReLU function."""
    return np.maximum(0, x)

def sample_tanh(x):
    """Sample a hyperbolic tangent function."""
    return np.tanh(x)

def sample_asin(x):
    """Sample an arcsine function."""
    return np.arcsin(np.clip(x, -1, 1))  # Clip x to avoid domain errors

def sample_acos(x):
    """Sample an arccosine function."""
    return np.arccos(np.clip(x, -1, 1))  # Clip x to avoid domain errors


def sample_synthetic_data(x, depth=1):
    functions = [
        sample_polynomial,
        sample_sine,
        # sample_tan,
        sample_exp,
        # sample_abs_log,
        # sample_relu,
        # sample_tanh,
        #sample_asin,
        #sample_acos
    ]
    for _ in range(depth):
        x = np.random.uniform(-10.0, 10.0) * x + np.random.uniform(-10.0, 10.0)
        values = []
        for j in range(np.random.randint(1, 4)):
            func = np.random.choice(functions)
            scale = np.random.uniform(-2.0, 2.0)
            offset = np.random.uniform(-1.0, 1.0)
            values.append(scale * func(x) + offset)

        values = np.array(values)
        operand = np.random.choice(['+', '*'])
        if operand == '+':
            x = np.sum(values, axis=0)
        elif operand == '*':
            x = np.prod(values, axis=0)
        elif operand == '/':
            x = values[0]
            for v in values[1:]:
                x /= (v)
        
    x /= np.max(np.abs(x))  # Normalize to the range [-1, 1]
    return x

def generator(context_length, prediction_length, num_values):
    for i in range(num_values):
        x = np.linspace(-10, 10, context_length + prediction_length)
        y = sample_synthetic_data(x, depth=4)
        y = y.reshape((context_length + prediction_length, 1)).astype(np.float32)
        yield {"target": y}

def create_tfrecords_dataset(data_dir: str, name: str, context_length=512, prediction_length=96, num_training_samples=1000000, num_validation_samples=100000, num_test_samples=100000):

    output_signature = {"target": tf.TensorSpec(shape=(context_length + prediction_length, 1), dtype=tf.float32)}
    
    ds_train = tf.data.Dataset.from_generator(partial(generator, context_length, prediction_length, num_training_samples), 
                                        output_signature=output_signature)
    
    ds_val = tf.data.Dataset.from_generator(partial(generator, context_length, prediction_length, num_validation_samples), 
                                        output_signature=output_signature)
    ds_test = tf.data.Dataset.from_generator(partial(generator, context_length, prediction_length, num_test_samples),
                                        output_signature=output_signature)    
    

    # Define the builder.
    single_number_builder = tfds.dataset_builders.TfDataBuilder(
        name=name,
        config=f"{context_length}_{prediction_length}_{num_training_samples}_{num_test_samples}",
        version="1.0.0",
        data_dir=data_dir,
        split_datasets={
            "train": ds_train,
            "val": ds_val,
            "test": ds_test,
        },
        features=tfds.features.FeaturesDict({
            "target": tfds.features.Tensor(shape=(context_length + prediction_length, 1), dtype=tf.float32),
        }),
        description="Test",
        release_notes={
            "1.0.0": "s",
        }
    )

    # Make the builder store the data as a TFDS dataset.
    single_number_builder.download_and_prepare()

    # save metadata
    metadata = {"normalization": (0, 1), #
                "length_train": num_training_samples, 
                "length_val": num_validation_samples, 
                "length_test": num_test_samples}
    with open(f"{data_dir}/{name}/{context_length}_{prediction_length}_{num_training_samples}_{num_test_samples}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


class SyntheticV2:
    def __init__(self, context_length, prediction_length, dataset_base_path, num_training_samples, num_test_samples, name):
        self.dataset_base_path = Path(dataset_base_path) / "syntheticV2"
        tf_name = f"tf_{name.lower()}"
        dataset_save_path_relative = f"{tf_name}/{context_length}_{prediction_length}_{num_training_samples}_{num_test_samples}"
        
        # if the dataset is not already generated, generate it
        if not os.path.exists(self.dataset_base_path / dataset_save_path_relative):
            print("Dataset not found as tfrecords. Generating tfrecords dataset")
            create_tfrecords_dataset(data_dir=self.dataset_base_path,
                                     name=tf_name,
                                     context_length=context_length, 
                                     prediction_length=prediction_length,
                                     num_training_samples=num_training_samples,
                                     num_validation_samples=num_test_samples,
                                     num_test_samples=num_test_samples)
            print("Created tfrecords file.")

        # load tfrecords dataset
        self.train_data_source = tfds.load(dataset_save_path_relative, split="train", data_dir=self.dataset_base_path)
        self.val_data_source = tfds.load(dataset_save_path_relative, split="val", data_dir=self.dataset_base_path)
        self.test_data_source = tfds.load(dataset_save_path_relative, split="test", data_dir=self.dataset_base_path)

        # load metadata
        metadata_file = self.dataset_base_path / dataset_save_path_relative / "metadata.pkl"
        print(metadata_file)
        print("dddddddddddddddddddddddddddddddd")
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        self.mean = metadata["normalization"][0]
        self.std = metadata["normalization"][1]
        self.length_train = metadata["length_train"]
        self.length_val = metadata["length_val"]
        self.length_test = metadata["length_test"]

        self.denormalize = partial(denormalize, mean=self.mean, std=self.std)
        self.num_features = 1

        self.init_dataloader_train = partial(init_tf_dataloader_timeseries, context_length=context_length)
        self.init_dataloader_val = partial(init_tf_dataloader_timeseries, context_length=context_length)
        self.init_dataloader_test = partial(init_tf_dataloader_timeseries, context_length=context_length)


if __name__ == "__main__":
    data_dir = f"/data/datasets/syntheticV2"
    name = "synthetic_v2"
    create_tfrecords_dataset(data_dir, name, context_length=512, prediction_length=96)
    # for i in range(100000):
    #     # Example usage
    #     x = np.linspace(-10, 10, 512 + 96)
    #     y = sample_synthetic_data(x, depth=4)

    #     import matplotlib.pyplot as plt
    #     plt.plot(x, y)
    #     plt.title("Sampled Synthetic Data")
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.grid()
    #     plt.show()
