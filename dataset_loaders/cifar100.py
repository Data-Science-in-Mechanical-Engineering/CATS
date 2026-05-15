from matplotlib import pyplot as plt
from pathlib import Path

from functools import partial

import tensorflow_datasets as tfds

from dataset_loaders.dataset_utils import init_tf_dataloader_image, denormalize
from dataset_loaders.rand_augment import init_randaugment


class Cifar100:
    def __init__(self, dataset_base_path, augment_magnitude, resolution, name):
        self.dataset_base_path = Path(dataset_base_path) / "cifar100"

        # load tfrecords dataset
        self.train_data_source = tfds.load(f"cifar100", split="train", data_dir=self.dataset_base_path)
        self.val_data_source = tfds.load(f"cifar100", split="test", data_dir=self.dataset_base_path)
        self.test_data_source = tfds.load(f"cifar100", split="test", data_dir=self.dataset_base_path)

        self.mean = 0
        self.std = 255
        self.length_train = 50000
        self.length_val = 10000
        self.length_test = 10000

        self.denormalize = partial(denormalize, mean=self.mean, std=self.std)
        self.num_features = 8

        augment_functions = init_randaugment(augment_magnitude)
        self.init_dataloader_train = partial(init_tf_dataloader_image, rand_augment_functions=augment_functions, resolution=resolution)
        self.init_dataloader_val = partial(init_tf_dataloader_image, rand_augment_functions=None, resolution=resolution)
        self.init_dataloader_test = partial(init_tf_dataloader_image, rand_augment_functions=None, resolution=resolution)


if __name__ == "__main__":
    dataset = Cifar100("/data/datasets", 0.5, 224, "cifar100")
    augment_functions = init_randaugment(0.0)
    ds_train = dataset.init_dataloader_train(dataset.train_data_source, 64, 1, 4)
    while True:
        batch = next(ds_train)


        fig, axes = plt.subplots(8, 8, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i >= len(batch["input"]):
                break
            print((batch["input"][i] + 1) / 2 * 255)
            image = (batch["input"][i] + 1) / 2 * 255
            ax.imshow(image.astype("uint8"))
            ax.axis("off")
        plt.tight_layout()
        plt.show()