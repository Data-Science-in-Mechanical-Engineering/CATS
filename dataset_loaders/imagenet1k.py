from matplotlib import pyplot as plt
from pathlib import Path

from functools import partial

import tensorflow_datasets as tfds

from dataset_loaders.dataset_utils import init_tf_dataloader_image, denormalize
from dataset_loaders.rand_augment import init_randaugment


class Imagenet1k:
    def __init__(self, dataset_base_path, augment_magnitude, resolution, num_parallel_calls, name):
        self.dataset_base_path = Path(dataset_base_path) / "imagenet1k"

        # load tfrecords dataset
        download_config = tfds.download.DownloadConfig(manual_dir=self.dataset_base_path)
        self.train_data_source = tfds.load(f"imagenet2012", split="train", data_dir=self.dataset_base_path, download_and_prepare_kwargs={"download_config": download_config})
        self.val_data_source = tfds.load(f"imagenet2012", split="validation", data_dir=self.dataset_base_path, download_and_prepare_kwargs={"download_config": download_config})
        self.test_data_source = tfds.load(f"imagenet2012", split="test", data_dir=self.dataset_base_path, download_and_prepare_kwargs={"download_config": download_config})

        self.mean = 0
        self.std = 255
        self.length_train = 1_281_167
        self.length_val = 50_000
        self.length_test = 100_000

        self.denormalize = partial(denormalize, mean=self.mean, std=self.std)
        self.num_features = 8

        augment_functions = init_randaugment(augment_magnitude)
        self.init_dataloader_train = partial(init_tf_dataloader_image, rand_augment_functions=augment_functions, resolution=resolution, num_parallel_calls=num_parallel_calls)
        self.init_dataloader_val = partial(init_tf_dataloader_image, rand_augment_functions=None, resolution=resolution, num_parallel_calls=num_parallel_calls)
        self.init_dataloader_test = partial(init_tf_dataloader_image, rand_augment_functions=None, resolution=resolution, num_parallel_calls=num_parallel_calls)


if __name__ == "__main__":
    dataset = Cifar100("/data/datasets", "cifar100")
    augment_functions = init_randaugment(0.2)
    ds_train = init_tf_dataloader_image(dataset.train_data_source, 64, 1, 42, augment_functions)
    while True:
        batch = next(ds_train)


        fig, axes = plt.subplots(8, 8, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i >= len(batch["input"]):
                break
            print(batch["input"][i]* 255)
            image = batch["input"][i] * 255
            ax.imshow(image.astype("uint8"))
            ax.axis("off")
        plt.tight_layout()
        plt.show()