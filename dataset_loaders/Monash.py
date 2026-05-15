from functools import partial
import os
from pathlib import Path
import pickle
from datasets import load_dataset


from datetime import datetime
from distutils.util import strtobool

import numpy as np
import pandas as pd

# import tensorflow_datasets as tfds

from dataset_loaders.dataset_utils import DataSource, init_tf_dataloader_timeseries, normalize, denormalize, SplitMovingWindowDataSource
from dataset_loaders.tf_records_dataset import TfSource, create_tfrecords_dataset


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


class MonashDataSource(DataSource):
    def __init__(self, name, context_length, prediction_length, stride, dataset_base_path):
        super().__init__(name, context_length, prediction_length, stride, 1, Path(dataset_base_path) / "monash" / name)

        loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length, = convert_tsf_to_dataframe(
            full_file_path_and_name=f"{self.dataset_base_path}/{name}.tsf",
            replace_missing_vals_with="NaN",
            value_column_name="series_value",
        )
        
        # assert contain_missing_values, "Timeseries contains missing values"

        data = []
        
        for i in range(len(loaded_data)):
            data.append(np.array([loaded_data["series_value"][i]]).T)

        data_train = data[0:int(len(data) * 0.75)]
        data_val = data[int(len(data) * 0.75):int(len(data) * 0.875)]
        data_test = data[int(len(data) * 0.875):]

        # normalize
        # print(data_train)
        data_train_total = np.concatenate(data_train, axis=0)
        self.mean = data_train_total.mean(axis=0)
        self.std = data_train_total.std(axis=0)

        data_train = [normalize(d, self.mean, self.std) for d in data_train]
        data_val = [normalize(d, self.mean, self.std) for d in data_val]
        data_test = [normalize(d, self.mean, self.std) for d in data_test]

        self.train_data_source = SplitMovingWindowDataSource(data_train, context_length, prediction_length, stride)
        self.val_data_source = SplitMovingWindowDataSource(data_val, context_length, prediction_length, stride)
        self.test_data_source = SplitMovingWindowDataSource(data_test, context_length, prediction_length, stride)

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


class Monash:
    def __init__(self, context_length, prediction_length, stride, dataset_base_path, name):
        self.dataset_base_path = Path(dataset_base_path) / "monash" / name

        tf_name = f"tf_{name.lower()}"
        # if the dataset is not already generated, generate it
        if not os.path.exists(self.dataset_base_path / f"{tf_name}/{context_length}_{prediction_length}_{stride}"):
            print("Dataset not found as tfrecords. Generating tfrecords dataset")
            icd_data_source = MonashDataSource(name, context_length, prediction_length, stride, self.dataset_base_path)
            create_tfrecords_dataset(data_source=icd_data_source,
                                     data_dir=self.dataset_base_path,
                                     name=tf_name)
            print("Created tfrecords file.")

        # load tfrecords dataset
        self.train_data_source = tfds.load(f"{tf_name}/{context_length}_{prediction_length}_{stride}", split="train", data_dir=self.dataset_base_path)
        self.val_data_source = tfds.load(f"{tf_name}/{context_length}_{prediction_length}_{stride}", split="val", data_dir=self.dataset_base_path)
        self.test_data_source = tfds.load(f"{tf_name}/{context_length}_{prediction_length}_{stride}", split="test", data_dir=self.dataset_base_path)

        # load metadata
        metadata_file = self.dataset_base_path / f"{tf_name}/metadata.pkl"
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        self.mean = metadata["normalization"]["mean"]
        self.std = metadata["normalization"]["std"]
        self.length_train = metadata["length_train"]
        self.length_val = metadata["length_val"]
        self.length_test = metadata["length_test"]

        self.denormalize = partial(denormalize, mean=self.mean, std=self.std)
        self.num_features = 1

        self.init_dataloader_train = partial(init_tf_dataloader_timeseries, context_length=context_length)
        self.init_dataloader_val = partial(init_tf_dataloader_timeseries, context_length=context_length)
        self.init_dataloader_test = partial(init_tf_dataloader_timeseries, context_length=context_length)


if __name__ == "__main__":
    # data_source = load_dataset("monash_tsf", "oikolab_weather")
    # train_source = data_source["train"]
    # for t in train_source:
    #     print(len(t["target"]))
    # monash = Monash(512, 96, 95, "/data/datasets", "london_smart_meters_dataset")

    name = "london_smart_meters_dataset"
    loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length, = convert_tsf_to_dataframe(
        full_file_path_and_name=f"/data/datasets/monash/{name}/{name}.tsf",
        replace_missing_vals_with="NaN",
        value_column_name="series_value",
    )

    #print(loaded_data["obs_or_fcst"].unique())

    import matplotlib.pyplot as plt
    print(len(loaded_data["series_value"][0]))
    # Plot the first series in the loaded data
    plt.plot(loaded_data["series_value"][0])
    plt.title("First Series Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()

    print(loaded_data)