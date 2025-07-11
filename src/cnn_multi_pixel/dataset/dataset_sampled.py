import numpy as np
import re
import os
import gzip
import pickle
from torch.utils.data import Dataset
from src.cnn_multi_pixel.helper_functions.subsampler import sample_raw_dataset

class TraceDatasetSampled(Dataset):
    ''' 
    Class used to create sampled datasets. Each dataset represents a single trace folder.
    Dataset values via __getitem__:
        Index: int; Full digital value.
        Item:  list of np.float32; Full list of trace values of corresponding digital value.
    Attributes:
        1) self.folder_name: string; Name of folder. Used to identify raw dataset.
        2) self.sample_trace_val: list; List of sampled trace values.
        3) self.sample_info: tuple;
            sample_info[0]: sample_interval
            sample_info[1]: sample_duration
            sample_info[2]: sample_mode; SINGLE sample mode
    Input for initialization:
        1) folder_name: string; Name of folder where traces originate
        2) label_dict; dictionary;
            Key: string; file name
            Value: list of tuples; np.float64; trace value array
    '''
    def __init__(self, folder_name, sample_trace_val, sample_info):
        self.folder_name        = folder_name
        self.sample_trace_val   = sample_trace_val
        self.sample_interval    = sample_info[0]
        self.sample_duration    = sample_info[1]
        self.sample_mode        = sample_info[2]
        self.len                = int(self.sample_duration / self.sample_interval)

    # Number of total files within folder
    def __len__(self):
        return self.len
    
    # Get item via digital value
    def __getitem__(self, digital_value):
        return self.sampled_dict[digital_value]
    
    # Get folder name
    def get_folder_name(self):
        return self.folder_name
    
    # Get full sampled_dict
    def get_sampled_dict(self):
        return self.sampled_dict
    
    # Get sampe_info tuple
    def get_sample_info(self):
        return self.sample_info
    
def create_and_save_sampled_datasets(raw_sampled_dir, raw_dataset_list, folder_dict):
    ''' 
    Function used to create and save sampled datasets for a list of raw datasets.
    Assumed to be run right after 'create_and_save_raw_dataset'
    Input:
        1) raw_sampled_dir: string; Full path to directory where all sampled datasets are stored as .gz files.
        2) raw_dataset_list: list; List of raw datasets to sample
        3) folder_dict: dictionary;
            Key: Folder name
            Value:
                folder_dict["sample_modes"] = sample modes; Possible to have mulitple modes specified
                folder_dict["sample_interval"] = sample interval
                folder_dict["sample_duration"] = sample duration
    Returns:
        1) new_sampled_datasets: list; List of newly created sampled datasets
    '''
    new_sampled_datasets = []
    for raw_dataset in raw_dataset_list:
        folder_name = raw_dataset.get_folder_name()
        sample_modes = folder_dict[folder_name]["sample_modes"]
        sample_interval = folder_dict[folder_name]["sample_interval"]
        sample_duration = folder_dict[folder_name]["sample_duration"]
        max_samples = int(sample_duration / sample_interval)
        for sample_mode in sample_modes:
            new_dataset = sample_raw_dataset(raw_dataset, sample_interval, max_samples, sample_mode)
            new_dataset_path = os.path.join(raw_sampled_dir, f"{raw_dataset.get_folder_name()}_{sample_mode}_sam.gz")
            save_data = {"folder_name": new_dataset.get_folder_name(), "sampled_dict": new_dataset.get_raw_dict()}
            with gzip.open(new_dataset_path, "wb") as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            new_sampled_datasets.append(new_dataset)
    return new_sampled_datasets

def load_sampled_dataset(sampled_save_dir, folder_list):
    ''' 
    Function used to load saved raw datasets.
    Input:
        1) sampled_save_dir: string; Full path to directory where all sampled datasets are stored as .gz files.
        2) folder_list: list of strings; List of all raw dataset folder names to load
    '''
    load_dataset_list = []
    for folder_name in folder_list:
        folder_pattern = rf"^{re.escape(folder_name)}_(\w+)_sam\.gz$"
        for filename in os.listdir(sampled_save_dir):
            match = re.match(folder_pattern, filename)
            if match:
                load_data_path = os.path.join(sampled_save_dir, filename)
                with gzip.open(load_data_path, "rb") as f:
                    load_data = pickle.load(f)
                    load_dataset = TraceDatasetSampled(load_data["folder_name"], load_data["sampled_dict"])
                load_dataset_list.append(load_dataset)
    return load_dataset_list