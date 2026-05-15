
import numpy as np
from dataset_loaders.utsdataset import UTSDataSource
import time

from torch.utils.data import Dataset, DataLoader, default_collate
import torch



class TorchDatasetWrapper(Dataset):
    def __init__(self, data_source):
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        return  {"target": torch.zeros((700, 9), dtype=torch.float32)}
    
def numpy_collate(batch):
    """
    Collate function specifies how to combine a list of data samples into a batch.
    default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
    """
    batch = default_collate(batch)
    return {k: np.asarray(v) for k, v in batch.items()}


if __name__ == "__main__":
    data_source = UTSDataSource(dataset_base_path="/data/datasets", 
                                subset_name=r'UTSD-12G', 
                                flag='train',
                                split=0.9, context_length=512, prediction_length=96,
                                scale=True, stride=1)
    # dataset_train = TorchDatasetWrapper(data_source.get_train_data_source())

    dataloader = DataLoader(data_source, 
                            batch_size=1024 * 16, 
                            shuffle=True, 
                            num_workers=31, 
                            prefetch_factor=None,
                            in_order=True)  #, collate_fn=numpy_collate)
    
    # dataloader = init_tf_dataloader_timeseries(data_source=TorchDatasetWrapper(data_source.get_train_data_source()),
    #                                                 batch_size=1024,
    #                                                 num_epochs=1,
    #                                                 num_features=9,
    #                                                 prediction_length=700,
    #                                                 context_length=0,
    #                                                 seed=1,
    #                                                 num_workers=31,
    #                                                 )

    # print(len(dataloader))
    # dataloader = iter(dataloader)

    # for d in tqdm.tqdm(dataloader, desc="Loading batches"):
    #     d["target"] = np.asarray(d["target"])
    #     # print(type(d["target"]))
    #     time.sleep(1.0)
    dataloader = iter(dataloader)
    while True:
        print("----------------")
        start_time = time.time()
        d = next(dataloader)
        start_time2 = time.time()
        d = {k: np.asarray(v) for k, v in d.items()}
        duration_total = (time.time() - start_time) * 1000
        duration_conversion = (time.time() - start_time2) * 1000
        print(f"Batch time: {duration_total:.4f} ms")
        print(f"Conversion time: {duration_conversion:.4f} ms")
        time.sleep(0.1)  # Sleep to simulate processing time


    