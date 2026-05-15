# file from: https://github.com/thuml/Large-Time-Series-Model/blob/main/scripts/UTSD/utsdataset.py

from functools import partial
import datasets
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dataset_loaders.dataset_utils import DataSource, denormalize


class UTSDataSource(Dataset):
    # from: https://github.com/thuml/Large-Time-Series-Model/blob/main/scripts/UTSD/utsdataset.py
    def __init__(self, dataset_base_path, subset_name=r'UTSD-1G', flag='train', split=0.9, context_length=None, prediction_length=None, scale=True, stride=1):
        self.dataset_base_path = dataset_base_path
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.seq_len = context_length + prediction_length
        assert flag in ['train', 'val']
        assert split >= 0 and split <=1.0
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.scale = scale
        self.split = split
        self.stride = stride

        self.data_list = []
        self.n_window_list = []

        self.subset_name = subset_name
        self.__read_data__()

    def __read_data__(self):
        dataset = datasets.load_dataset("thuml/UTSD", self.subset_name, split='train', cache_dir=self.dataset_base_path)
        # split='train' contains all the time series, which have not been divided into splits, 
        # you can split them by yourself, or use our default split as train:val = 9:1
        print('Indexing dataset...')
        counter = 0
        for item in tqdm(dataset):
            # if counter > 10000:
            #     break
            counter += 1
            self.scaler = StandardScaler()
            data = item['target']
            data = np.array(data).reshape(-1, 1)
            num_train = int(len(data) * self.split)
            border1s = [0, num_train- self.context_length]  #  - self.context_length
            border2s = [num_train, len(data)]

            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.scale:
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = self.scaler.transform(data)

            data = data[border1:border2]
            n_window = (len(data) - self.seq_len) // self.stride + 1
            # n_windows_temp = (border2s[1] - border1s[0] - self.prediction_length - self.seq_len) // self.stride + 1
            if n_window < 1 or border1s[1] <=0:  #  or n_windows_temp < 1:
                continue
            # if i < 5:
            # print(np.mean(data))
            # print(np.std(data))
            # plt.figure(figsize=(10, 4))
            # plt.plot(orig_data, color="blue")
            # plt.title(f"Time Series Sample {i}")
            # plt.xlabel("Time Step")
            # plt.ylabel("Normalized Value")
            # plt.grid(True)
            # plt.show()

            self.data_list.append(data)
            self.n_window_list.append(n_window if len(self.n_window_list) == 0 else self.n_window_list[-1] + n_window)

        self.indexes = np.array(self.n_window_list)
        print(len(self.indexes), "time series windows in total")

    def __getitem__(self, index):
        # you can wirte your own processing code here
        # dataset_index = 0
        # while index >= self.n_window_list[dataset_index]:
        #     dataset_index += 1        
        dataset_index = np.searchsorted(self.indexes, index, side="right")

        index = index - self.n_window_list[dataset_index - 1] if dataset_index > 0 else index
        n_timepoint = (len(self.data_list[dataset_index]) - self.seq_len) // self.stride + 1
        
        s_begin = index % n_timepoint
        s_begin = self.stride * s_begin
        input_end = s_begin + self.context_length
        target_end = s_begin + self.seq_len

        # seq_x = self.data_list[dataset_index][s_begin:input_end, :]
        seq_y = self.data_list[dataset_index][s_begin:target_end, :]

        return {"target": seq_y}  #, "input": seq_x}
    
    def __len__(self):
        return self.n_window_list[-1]
    

class UTSDataset(DataSource):
    """
    UTS dataset loader.
    This class loads the UTS dataset and provides access to the time series data.
    """
    def __init__(self, name=None, context_length=None, prediction_length=None, stride=None, dataset_base_path=None, subset_name=r'UTSD-1G', split=0.9):
        super().__init__(name=name, context_length=context_length, prediction_length=prediction_length, stride=stride, num_features=1, dataset_base_path=dataset_base_path)
        self.subset_name = subset_name

        self.train_data_source = UTSDataSource(dataset_base_path=dataset_base_path,
                                               subset_name=subset_name, 
                                               flag='train', 
                                               split=split, 
                                               context_length=context_length, 
                                               prediction_length=prediction_length, 
                                               scale=True, 
                                               stride=stride)
        
        self.val_data_source = UTSDataSource(dataset_base_path=dataset_base_path,
                                             subset_name=subset_name, 
                                             flag='val', 
                                             split=split, 
                                             context_length=context_length, 
                                             prediction_length=prediction_length, 
                                             scale=True, 
                                             stride=stride)
        self.test_data_source = self.val_data_source
        
        self.mean = 0
        self.std = 1
        self.denormalize = partial(denormalize, mean=self.mean, std=self.std)

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




# See ```download_dataset.py``` to download the dataset first
if __name__ == '__main__':
    # dataset = UTSDataset(subset_name=r'UTSD-1G', input_len=672, output_len=0, flag='train')
    dataset = UTSDataset(subset_name=r'UTSD-1G', input_len=720, output_len=96, flag='train')
    print(f'total {len(dataset)} time series windows (sentence)')