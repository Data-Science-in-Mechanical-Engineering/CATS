import pandas as pd
from pathlib import Path
import numpy as np

from dataset_loaders.dataset_utils import normalize, denormalize, SplitMovingWindowDataSource

from torch.utils.data import Dataset
from dataset_loaders.dataset_utils import DataSource


class ECLDataSource(DataSource):
    def __init__(self, context_length, prediction_length, dataset_base_path, name):
        super().__init__(name, context_length, prediction_length, 1, 1, Path(dataset_base_path) / "ECL")
        self.name = name
        
        # Load the single CSV file
        csv_file = self.dataset_base_path / "electricity.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_file}")
        
        # Load data and extract only the OT column
        df = pd.read_csv(csv_file)
        ot_data = df['OT'].values.reshape(-1, 1)  # Reshape to be 2D (timesteps, 1 feature)
        
        # Split data: first 70% train, next 15% val, last 15% test
        total_length = len(ot_data)
        train_end = int(0.7 * total_length)
        val_end = int(0.85 * total_length)
        
        data_train = [ot_data[:train_end]]
        data_val = [ot_data[train_end:val_end]]
        data_test = [ot_data[val_end:]]

        # normalize using training data statistics
        data_train_total = np.concatenate(data_train, axis=0)
        self.mean = data_train_total.mean(axis=0)
        self.std = data_train_total.std(axis=0)

        data_train = [normalize(d, self.mean, self.std) for d in data_train]
        data_val = [normalize(d, self.mean, self.std) for d in data_val]
        data_test = [normalize(d, self.mean, self.std) for d in data_test]

        self.train_data_source = SplitMovingWindowDataSource(data_train, context_length, prediction_length)
        self.val_data_source = SplitMovingWindowDataSource(data_val, context_length, prediction_length)
        self.test_data_source = SplitMovingWindowDataSource(data_test, context_length, prediction_length)

        self.num_features = 1

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


if __name__ == "__main__":
    # Test the ECLDataSource
    ecl_data_source = ECLDataSource(512, 128, '/data/datasets', 'ECL')
    print(f"Number of features: {ecl_data_source.num_features}")
    print(f"Train dataset length: {len(ecl_data_source.train_data_source)}")
    print(f"Val dataset length: {len(ecl_data_source.val_data_source)}")
    print(f"Test dataset length: {len(ecl_data_source.test_data_source)}")
    
    # Test a sample
    sample = ecl_data_source.train_data_source[0]
    print(f"Sample shape: {sample['target'].shape}")
    print(f"Normalization params - Mean: {ecl_data_source.mean}, Std: {ecl_data_source.std}")
    
    # Test ECL class
    # ecl = ECL(512, 128, '/data/datasets', 'ECL')
    # print(f"ECL class features: {ecl.num_features}")