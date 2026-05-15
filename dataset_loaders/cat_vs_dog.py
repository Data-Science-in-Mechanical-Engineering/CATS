import grain.python as grain
import pandas as pd
from pathlib import Path
import numpy as np

from functools import partial
from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from dataset_loaders.dataset_utils import normalize, denormalize
from dataset_loaders.dataset_utils import DataSource

from PIL import Image
import random


class CatVsDogSingle(Dataset):
    def __init__(self, resolution, dataset_base_path, image_begin_idx, image_end_idx, augment=False, fileformat=".jpg", test=False):
        self.__resolution = resolution
        self.__image_begin_idx = image_begin_idx
        self.__image_end_idx = image_end_idx

        if augment:
            self.data_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(degrees=90),
                v2.RandomResizedCrop(size=(self.__resolution, self.__resolution), antialias=True),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                v2.Grayscale(num_output_channels=1),
                v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.data_transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(self.__resolution, self.__resolution), antialias=True),
                v2.Grayscale(num_output_channels=1),
                v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
                # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                v2.Normalize(mean=[0.5], std=[0.5])
            ])

        
        self.__data = [{"image": None, "target": None} for _ in range((image_end_idx - image_begin_idx)*2)]
        for i, animal in enumerate(["Cat", "Dog"]):
            for idx in tqdm(range(image_begin_idx, image_end_idx), desc=f"Loading {animal} images"):
                image_path = Path(dataset_base_path) / f"Cat_vs_Dog/{'Test/' if test else ''}{animal}/{idx}{fileformat}"
                image = np.asarray(Image.open(image_path).convert("RGB"))
                self.__data[idx + i*(image_end_idx - image_begin_idx) - image_begin_idx]["image"] = image
                self.__data[idx - i*(image_end_idx - image_begin_idx) - image_begin_idx]["target"] = 0 if animal == "Cat" else 1

                if image.shape[-1] != 3:
                    print(image_path)
                    

    def __getitem__(self, idx):
        data = self.__data[idx]
        # Change from CxHxW to HxWxC
        return  {"image": self.data_transform(data["image"]).permute(1, 2, 0), "target": data["target"]}

    def __len__(self):
        return len(self.__data)

class CatVsDog(DataSource):
    def __init__(self, resolution, dataset_base_path, name):
        self.__mean = 0.0
        self.__std = 1.0

        num_images = 12499
        val_border = int(num_images * 0.8)

        self.train_data_source = CatVsDogSingle(resolution=resolution,
                                                dataset_base_path=dataset_base_path,
                                                image_begin_idx=0,
                                                image_end_idx=val_border,
                                                augment=True)
        self.val_data_source = CatVsDogSingle(resolution=resolution,
                                              dataset_base_path=dataset_base_path,
                                              image_begin_idx=val_border,
                                              image_end_idx=num_images,
                                              augment=False)
        self.test_data_source = CatVsDogSingle(resolution=resolution,
                                              dataset_base_path=dataset_base_path,
                                              image_begin_idx=0,
                                              image_end_idx=1,
                                              augment=False,
                                              fileformat=".png",
                                              test=True)

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
    
if __name__ == "__main__":
    ds = CatVsDogSingle(224, "/data/datasets", 0, 100, augment=False)
    print(len(ds))

    import matplotlib.pyplot as plt

    # Randomly select 16 indices
    indices = random.sample(range(len(ds)), 16)

    # Create a 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        sample = ds[idx]
        img = sample["image"]
        label = "Cat" if sample["target"] == 0 else "Dog"
        print(img.shape)
        # Denormalize the image for display
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img)
        axes[i].set_title(label)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()